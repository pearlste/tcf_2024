#! /usr/bin/python

# Usage:
#   matte_and_scale.py <src_dir> <dst_dir>
#

import sys
import os
import re
import subprocess
import numpy as np
import argparse

import    argparse

parser  = argparse.ArgumentParser(description='Example:\n    ./prepare_imagenet.py ~pearlstl/image-net ~pearlstl/image-net_256x256')

parser.add_argument('src_dir', help='Path to source directory')
parser.add_argument('dst_dir', help='Path to destination directory')

args        = parser.parse_args()
src_dir     = args.src_dir
dst_dir     = args.dst_dir

os.chdir(src_dir)
file_list = os.listdir(".")

filenum = 0
for the_file in file_list:
    pattern = "(.*)\.(JPEG|JPG|PNG|BMP)"
    m = re.search( pattern, the_file, re.I )
    if m:
        file_root = m.group(1)
        sys_cmd = "file %s" % the_file
        try:

            the_line = subprocess.getoutput( [sys_cmd] )
        except subprocess.CalledProcessError as e:
            the_line = e.output
            #print "***** Caught CalledProcessorError exception in 'file' command: %s *****" % the_line
        # parse file size from:
        #   n02088466_1015.JPEG: JPEG image data, JFIF standard 1.01, aspect ratio, density 1x1, segment length 16, baseline, precision 8, 240x184, frames 3
        #pattern = "JFIF\ standard\ 1\.01.*precision.*,\s*([0-9]*)x([0-9]*),.*frames"
        pattern = ".*image\ data.*,\s*([0-9]+)x([0-9]+)"
        #print( str(the_line) )
        m = re.search( pattern, the_line )
        if not m:
            print ("Error: couldn't find %s in %s" % (pattern, the_line))
            #sys.exit([-1])
        else:
            wid = int(m.group(1))
            hgt = int(m.group(2))
            #print ("Found %d(wid) x %d(hgt) in %s" % (wid, hgt, the_line))
            #print "%24s: %4d x %4d" % (the_file, wid, hgt)
            
            # Need to pad width to match height
            if wid < hgt:
                # Indent from top, crop vertical
                out_wid = hgt
                out_hgt = hgt
                win_y   = 0
                win_x   = (hgt - wid) / 2
            else:
                # pad height to match width
                out_wid = wid
                out_hgt = wid
                win_y   = (wid - hgt) / 2
                win_x   = 0
            # crop=outw:outh:x:y
            #-vf "pad=width=1280:height=720:x=0:y=1:color=black"

            sys_cmd = 'ffmpeg -y -loglevel quiet -i %s -vf "pad=width=%d:height=%d:x=%d:y=%d:color=0x808080" -s 128x128 %s/%s.png' % (the_file, out_wid, out_hgt, win_x, win_y, dst_dir, file_root)
            #print( sys_cmd )
            
            try:

                the_line = subprocess.check_output( [sys_cmd], shell=True)
                print( "response: %s" % the_line )
            except subprocess.CalledProcessError as e:
                the_line = e.output
                print( "***** Caught CalledProcessorError exception in 'avconv' command: %s, output: %s *****" % (sys_cmd, the_line) )
            
    filenum = filenum + 1
    if (filenum % 10) == 0:
        print( "Done %d files" % filenum )

        
