import argparse
import logging
import os
import sys
import urllib2
import uuid
from mimetypes import guess_extension
from os.path import join as pjoin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='download file tool')
    parser.add_argument('-u', type=str, dest='url', help='url')
    parser.add_argument('-o', type=str, dest='output', default="./", help='output path')
    parser.add_argument('-n', type=str, dest='name', help='output name')
    args = parser.parse_args()
    if args.url is None:
        parser.print_help()
        sys.exit(2)

    #dlfile = urllib2.urlopen("http://i.chanpin100.com/151011138303399912")
    try:
        dlfile = urllib2.urlopen(args.url)
        extension = guess_extension(dlfile.info()['Content-Type'])
        if args.name is not None:
            tempname = args.name + extension
        else:
            tempname = str(uuid.uuid4()) + extension
        tempfile = pjoin(args.output, tempname)

        with open(tempfile,'wb') as output:
            output.write(dlfile.read())
        logging.info("download %r to %r", args.url, tempfile)
    except Exception as exc:
        print exc

