#!/usr/local/bin/python
# -*- coding: utf-8 -*-
################################################################################
# export_tflite.py
# Description:  converter from savedmodel to tflite.
#
# Author:       shozo fukuda
# Date:         Wed Jun 21 03:02:50 2023
# Last revised: $Date$
# Application:  Python 2.7
################################################################################

#<IMPORT>
import os,sys
import argparse

#<SUBROUTINE>###################################################################
# Function:     convert to tflite
# Description:  
# Dependencies: 
################################################################################
def to_tflite(savedmodel, name, signature):
    print('Loading savedmodel from "%s"...\n' % savedmodel)
    model = tf.saved_model.load(savedmodel)
    if signature == 'mapping':
        name += ".mapping.tflite"
        concrete_func = model.signatures['mapping']
        concrete_func.inputs[0].set_shape([1, None])
    elif signature == 'synthesis':
        name += ".synthesis.tflite"
        concrete_func = model.signatures['synthesis']
        concrete_func.inputs[0].set_shape([1, None, None])
    else:
        name += ".tflite"
        concrete_func = model.signatures['serving_default']
        concrete_func.inputs[0].set_shape([1, None])

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    tflite = converter.convert()

    print("Saving as tflite: %s" % name)
    with open(name, 'wb') as f:
        f.write(tflite)

#<TEST>#########################################################################
# Function:     command line
# Description:  
# Dependencies: 
################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export savedmodel to tflite")
    parser.add_argument('indir', help="savedmodel directory")
    parser.add_argument('name', nargs='?', default=None,
        help="tflite file (pass me without suffix) [default: basename of indir]")
    parser.add_argument('-s', '--signature', choices=['default', 'mapping', 'synthesis'], default='default',
        help="choose signature [default: default]")
    args = parser.parse_args()
    # args.src, args.dst, args.opt

    if args.name == None:
        args.name = os.path.basename(args.indir)

    if os.path.isdir(args.indir):
        print("Setup Tensorflow...")
        import tensorflow as tf

        to_tflite(args.indir, args.name, args.signature)
    else:
        print("Error: not exist the savedmodel directory: %s" %(args.indir))

# export_tflite.py
