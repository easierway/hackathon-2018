#!/usr/bin/env python

import sys
import urllib2
import re
import commands
import sys
import argparse
import time

image_regex = re.compile('<div class="pic" align="center">'
                         '<img src="(.*?)"/ width="80%"></div>', re.M)

title_regex = re.compile('<title>(.*?)</title>', re.M)


def title_image(url_address, prefix):
    html = urllib2.urlopen(url_address).read()
    image = image_regex.findall(html)
    title = title_regex.findall(html)

    if len(image) == 0 or len(title) == 0:
        return ''
    return 'wget {} -O "oimages/{}.jpg"'.format(image[0], prefix)
    # return 'wget {1} -O "oimages/{2}_{0}.jpg"'.format(title[0], image[0], prefix)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sidx", nargs="?", default=167, help="start idx")
    parser.add_argument("eidx", nargs="?", default=2000, help="end idx")
    args = parser.parse_args()
    for i in range(int(args.sidx), int(args.eidx)):
        time.sleep(1)
        try:
            command = title_image(
                'http://dili.bdatu.com/index.php/Share/index/id/' + str(i), str(i))
            if not command:
                print("address not valid. idx: [{}]".format(i))
                continue
            (status, output) = commands.getstatusoutput(command)
            if (status != 0):
                print("execute failed. command: [{}], status: [{}], output: [{}]".format(
                    command, status, output))
            else:
                print("execute success. command: [{}]".format(command))
        except Exception as e:
            print(e)


if __name__ == '__main__':
    main()
