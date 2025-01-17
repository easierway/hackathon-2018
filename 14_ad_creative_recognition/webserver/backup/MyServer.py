import os, sys, platform
import posixpath
import BaseHTTPServer
from SocketServer import ThreadingMixIn
import threading
import urllib, urllib2
import cgi
import shutil
import mimetypes
import re
import time
import urllib

from image_binary_classification import *

default_combined_model.load_weights()

try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO


def get_ip_address( ifname ):
    import socket
    import fcntl
    import struct
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return socket.inet_ntoa(fcntl.ioctl(
        s.fileno(),
        0x8915,  # SIOCGIFADDR
        struct.pack('256s', ifname[:15])
    )[20:24])


class GetWanIp:
    def getip( self ):
        myip = "127.0.0.1"
        return myip

    def visit( self, url ):

        opener = urllib2.urlopen(url, None, 3)
        if url == opener.geturl():
            str = opener.read()
        return re.search('(\d+\.){3}\d+', str).group(0)


def showTips( ):
    print ""
    print '----------------------------------------------------------------------->> '
    try:
        port = int(sys.argv[1])
    except Exception, e:
        print '-------->> Warning: Port is not given, will use deafult port: 8080 '
        print '-------->> if you want to use other port, please execute: '
        print '-------->> python SimpleHTTPServerWithUpload.py port '
        print "-------->> port is a integer and it's range: 1024 < port < 65535 "
        port = 8080

    if not 1024 < port < 65535: port = 8080
    # serveraddr = ('', port)
    print '-------->> Now, listening at port ' + str(port) + ' ...'
    osType = platform.system()
    if osType == "Linux":
        print '-------->> You can visit the URL:   http://' + GetWanIp().getip() + ':' + str(port)
    else:
        print '-------->> You can visit the URL:   http://127.0.0.1:' + str(port)
    print '----------------------------------------------------------------------->> '
    print ""
    return ('', port)


serveraddr = showTips()


def sizeof_fmt( num ):
    for x in ['bytes', 'KB', 'MB', 'GB']:
        if num < 1024.0:
            return "%3.1f%s" % (num, x)
        num /= 1024.0
    return "%3.1f%s" % (num, 'TB')


def modification_date( filename ):
    # t = os.path.getmtime(filename)
    # return datetime.datetime.fromtimestamp(t)
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(filename)))


class SimpleHTTPRequestHandler(BaseHTTPServer.BaseHTTPRequestHandler):
    """Simple HTTP request handler with GET/HEAD/POST commands.

    This serves files from the current directory and any of its
    subdirectories. The MIME type for files is determined by
    calling the .guess_type() method. And can reveive file uploaded
    by client.

    The GET/HEAD/POST requests are identical except that the HEAD
    request omits the actual contents of the file.

    """

    server_version = "SimpleHTTPWithUpload/"

    def do_GET( self ):
        """Serve a GET request."""
        # print "....................", threading.currentThread().getName()
        mpath, margs = urllib.splitquery(self.path)
        print (mpath, margs)
        print type(mpath)
        if mpath.find('correct') and margs != None:
            dst = '1'
            file_name = margs[:margs.find('=')]
            if margs[-1] == '1':
                dst = '0'

            shutil.move("./"+file_name,"./"+dst+'/'+file_name)
            f = StringIO()
            f.write('<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">')
            f.write("<html>\n<title>Corrected</title>\n")
            f.write("<body>\n<h2>Thanks for your correction</h2>\n")
            f.write("<br><a href=\"%s\">back</a>" % self.headers['referer'])
            f.write("<br><small>Powered By: caichao@amazon.com.</small>")
            f.write("</body></html>")
            length = f.tell()
            f.seek(0)
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.send_header("Content-Length", str(length))
            self.end_headers()
            if f:
                self.copyfile(f, self.wfile)
                f.close()
        else:
            f = self.send_head()
            if f:
                self.copyfile(f, self.wfile)
                f.close()

    def do_HEAD( self ):
        """Serve a HEAD request."""
        f = self.send_head()
        if f:
            f.close()

    def do_POST( self ):
        """Serve a POST request."""
        decision = 'Unknown'
        r, info,img_file, prediction = self.deal_post_data()
        print ("prediction",prediction)
        if (prediction > 0.5):
            decision = 'REJECTED'
        elif (prediction <= 0.5):
            decision = "PASSED"
        print r, info, "by: ", self.client_address
        f = StringIO()
        f.write('<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">')
        f.write("<html>\n<title>Upload Result Page</title>\n")
        f.write("<body>\n<h2>Upload Result Page</h2>\n")
        f.write('<img src='+img_file+' height=100 width=100 >\n')
        f.write("<hr>\n")
        if r:
            f.write("<strong>Success:</strong>")
        else:
            f.write("<strong>Failed:</strong>")
        f.write("<h3><strong> The decision is <red>"+decision+"</red></strong></h3>")
        f.write(info)
        request_params = "correct?"+img_file[1:]+"="+str(1 if prediction>0.5 else 0)
        print request_params
        f.write("<br><a href='"+self.headers['referer']+request_params+"'>Decision is wrong.</a>")
        f.write("<br><a href=\"%s\">back</a>" % self.headers['referer'])
        f.write("<br><small>Powered By: caichao@amazon.com.")

        f.write("</small></body>\n</html>\n")
        length = f.tell()
        f.seek(0)
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        if f:
            self.copyfile(f, self.wfile)
            f.close()

    def deal_post_data( self ):
        prediction = -1
        boundary = self.headers.plisttext.split("=")[1]
        remainbytes = int(self.headers['content-length'])
        line = self.rfile.readline()
        remainbytes -= len(line)
        if not boundary in line:
            return (False, "Content NOT begin with boundary","",prediction)
        line = self.rfile.readline()
        remainbytes -= len(line)
        fn = re.findall(r'Content-Disposition.*name="file"; filename="(.*)"', line)
        if not fn:
            return (False, "Can't find out file name...",fn[0],prediction)
        path = self.translate_path(self.path)
        osType = platform.system()
        try:
            if osType == "Linux":
                fn = os.path.join(path, fn[0].decode('gbk').encode('utf-8'))
            else:
                fn = os.path.join(path, fn[0])
        except Exception, e:
            return (False, "Please don't use Chinese file name.", fn[0],prediction)
        while os.path.exists(fn):
            fn += "_"
        line = self.rfile.readline()
        remainbytes -= len(line)
        line = self.rfile.readline()
        remainbytes -= len(line)
        try:
            out = open(fn, 'wb')
        except IOError:
            return (False, "Can't create file to write, do you have permission to write?",fn,prediction)

        preline = self.rfile.readline()
        remainbytes -= len(preline)
        while remainbytes > 0:
            line = self.rfile.readline()
            remainbytes -= len(line)
            if boundary in line:
                preline = preline[0:-1]
                if preline.endswith('\r'):
                    preline = preline[0:-1]
                out.write(preline)
                out.close()
                print (fn)
                pos = fn.rindex('/')
                print (pos,len(fn),fn[pos:])
                prediction = predict_by_default_model(fn)
                print ("p:",prediction)
                return (True, "File '%s' upload success!" % fn, fn[fn.rindex('/'):],prediction)
            else:
                out.write(preline)
                preline = line
        return (False, "Unexpect Ends of data.",fn[0],prediction)

    def send_head( self ):
        """Common code for GET and HEAD commands.

        This sends the response code and MIME headers.

        Return value is either a file object (which has to be copied
        to the outputfile by the caller unless the command was HEAD,
        and must be closed by the caller under all circumstances), or
        None, in which case the caller has nothing further to do.

        """
        path = self.translate_path(self.path)
        f = None
        if os.path.isdir(path):
            if not self.path.endswith('/'):
                # redirect browser - doing basically what apache does
                self.send_response(301)
                self.send_header("Location", self.path + "/")
                self.end_headers()
                return None
            for index in "index.html", "index.htm":
                index = os.path.join(path, index)
                if os.path.exists(index):
                    path = index
                    break
            else:
                return self.list_directory(path)
        ctype = self.guess_type(path)
        try:
            # Always read in binary mode. Opening files in text mode may cause
            # newline translations, making the actual size of the content
            # transmitted *less* than the content-length!
            f = open(path, 'rb')
        except IOError:
            self.send_error(404, "File not found")
            return None
        self.send_response(200)
        self.send_header("Content-type", ctype)
        fs = os.fstat(f.fileno())
        self.send_header("Content-Length", str(fs[6]))
        self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
        self.end_headers()
        return f

    def list_directory( self, path ):
        """Helper to produce a directory listing (absent index.html).

        Return value is either a file object, or None (indicating an
        error). In either case, the headers are sent, making the
        interface the same as for send_head().

        """
        try:
            list = os.listdir(path)
        except os.error:
            self.send_error(404, "No permission to list directory")
            return None
        list.sort(key=lambda a: a.lower())
        f = StringIO()
        displaypath = cgi.escape(urllib.unquote(self.path))
        f.write('<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">')
        f.write("<html>\n<title>Directory listing for %s</title>\n" % displaypath)
        f.write("<body>\n<h2>Directory listing for %s</h2>\n" % displaypath)
        f.write("<hr>\n")
        f.write("<form ENCTYPE=\"multipart/form-data\" method=\"post\">")
        f.write("<input name=\"file\" type=\"file\"/>")
        f.write("<input type=\"submit\" value=\"upload\"/>")
        f.write("              ")
        f.write("<input type=\"button\" value=\"HomePage\" onClick=\"location='/'\">")
        f.write("</form>\n")
        f.write("<hr>\n<ul>\n")
        for name in list:
            fullname = os.path.join(path, name)
            colorName = displayname = linkname = name
            # Append / for directories or @ for symbolic links
            if os.path.isdir(fullname):
                colorName = '<span style="background-color: #CEFFCE;">' + name + '/</span>'
                displayname = name
                linkname = name + "/"
            if os.path.islink(fullname):
                colorName = '<span style="background-color: #FFBFFF;">' + name + '@</span>'
                displayname = name
                # Note: a link to a directory displays with @ and links with /
            filename = os.getcwd() + '/' + displaypath + displayname
            f.write(
                '<table><tr><td width="60%%"><a href="%s">%s</a></td><td width="20%%">%s</td><td width="20%%">%s</td></tr>\n'
                % (urllib.quote(linkname), colorName,
                   sizeof_fmt(os.path.getsize(filename)), modification_date(filename)))
        f.write("</table>\n<hr>\n</body>\n</html>\n")
        length = f.tell()
        f.seek(0)
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        return f

    def translate_path( self, path ):
        """Translate a /-separated PATH to the local filename syntax.

        Components that mean special things to the local file system
        (e.g. drive or directory names) are ignored. (XXX They should
        probably be diagnosed.)

        """
        # abandon query parameters
        path = path.split('?', 1)[0]
        path = path.split('#', 1)[0]
        path = posixpath.normpath(urllib.unquote(path))
        words = path.split('/')
        words = filter(None, words)
        path = os.getcwd()
        for word in words:
            drive, word = os.path.splitdrive(word)
            head, word = os.path.split(word)
            if word in (os.curdir, os.pardir): continue
            path = os.path.join(path, word)
        return path

    def copyfile( self, source, outputfile ):
        """Copy all data between two file objects.

        The SOURCE argument is a file object open for reading
        (or anything with a read() method) and the DESTINATION
        argument is a file object open for writing (or
        anything with a write() method).

        The only reason for overriding this would be to change
        the block size or perhaps to replace newlines by CRLF
        -- note however that this the default server uses this
        to copy binary data as well.

        """
        shutil.copyfileobj(source, outputfile)

    def guess_type( self, path ):
        """Guess the type of a file.

        Argument is a PATH (a filename).

        Return value is a string of the form type/subtype,
        usable for a MIME Content-type header.

        The default implementation looks the file's extension
        up in the table self.extensions_map, using application/octet-stream
        as a default; however it would be permissible (if
        slow) to look inside the data to make a better guess.

        """

        base, ext = posixpath.splitext(path)
        if ext in self.extensions_map:
            return self.extensions_map[ext]
        ext = ext.lower()
        if ext in self.extensions_map:
            return self.extensions_map[ext]
        else:
            return self.extensions_map['']

    if not mimetypes.inited:
        mimetypes.init()  # try to read system mime.types
    extensions_map = mimetypes.types_map.copy()
    extensions_map.update({
        '': 'application/octet-stream',  # Default
        '.py': 'text/plain',
        '.c': 'text/plain',
        '.h': 'text/plain',
    })


class ThreadingServer(ThreadingMixIn, BaseHTTPServer.HTTPServer):
    pass


def test( HandlerClass=SimpleHTTPRequestHandler,
          ServerClass=BaseHTTPServer.HTTPServer ):
    BaseHTTPServer.test(HandlerClass, ServerClass)


if __name__ == '__main__':
    # test()

    # Single Thread
    # srvr = BaseHTTPServer.HTTPServer(serveraddr, SimpleHTTPRequestHandler)

    # Multi Thread
    srvr = ThreadingServer(serveraddr, SimpleHTTPRequestHandler)

    srvr.serve_forever()
