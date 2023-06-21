#!/usr/local/bin/python
# -*- coding: utf-8 -*-
################################################################################
# pkl2savedmodel.py
# Description:  converter pickle to savedmodel
#
# Author:       shozo fukuda
# Date:         Tue Jun 20 10:14:58 2023
# Last revised: $Date$
# Application:  Python 3
################################################################################

#<IMPORT>
import os
import shutil
import argparse
import pickle

#<SUBROUTINE>###################################################################
# Function:     convert pickle to savedmodel
# Description:  
# Dependencies: 
################################################################################
def to_savedmodel(pkl, outdir):
    # Load pretrained networks
    print('Loading networks from "%s"...' % pkl)
    with dnnlib.util.open_url(pkl) as fp:
        _G, _D, rGs = pickle.load(fp)

    # Config for saved_model
    Gs_args = rGs.static_kwargs.copy()
    Gs_args['num_fp16_res']    = 0
    Gs_args['randomize_noise'] = False
    Gs_args['return_dlatents'] = True

#    with tf.Graph().as_default(), tflib.create_session(force_as_default=True) as sess:
    with tflib.create_session(force_as_default=True) as sess:
        # Construct network
        Gs = tflib.Network(
            "Gs",
            func_name="training.networks.G_main",
            **Gs_args)
        Gs.copy_vars_from(rGs)

        # Get inputs/outputs
        [latents, _labels] = Gs.input_templates
        [images, dlatents] = Gs.output_templates

        # Save as saved_model
        builder = tf1.saved_model.Builder(outdir)
        builder.add_meta_graph_and_variables(
            sess,
            tags=["serve"],
            signature_def_map={
                tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf1.saved_model.predict_signature_def(
                    inputs={"latents": latents},
                    outputs={"images":  images}),
                "mapping": tf1.saved_model.predict_signature_def(
                    inputs={"latents": latents},
                    outputs={"dlatents": dlatents}),
                "synthesis": tf1.saved_model.predict_signature_def(
                    inputs={"dlatents": dlatents},
                    outputs={"images": images})
            })

        print("Saving as SavedModel: %s" %(outdir))
        builder.save()

#<TEST>#########################################################################
# Function:     command line
# Description:  
# Dependencies: 
################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert pretrained pickle to savedmodel")
    parser.add_argument('pkl', help="pickle file")
    parser.add_argument('outdir', help="saved_model direcotry")
    parser.add_argument('-f', '--force', action='store_true',
        help="remove outdir if existed")
    args = parser.parse_args()

    if args.force:
        shutil.rmtree(args.outdir, ignore_errors=True)
    elif os.path.isdir(args.outdir):
        print("Error: directory '%s' is already exist." %(args.outdir))
        exit()

    # Setup Tensorflow for legacy v1
    print("Setup Tensorflow...")
    import tensorflow as tf
    import tensorflow.compat.v1 as tf1
    tf1.logging.set_verbosity(tf1.logging.ERROR)
    tf1.disable_v2_behavior()
    tf1.enable_resource_variables()

    import dnnlib
    import dnnlib.tflib as tflib
    tflib.init_tf()

    # Convert
    to_savedmodel(args.pkl, args.outdir)

# pkl2savedmodel.py
