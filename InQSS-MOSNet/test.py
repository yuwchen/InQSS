#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time 
import numpy as np
import argparse
import fnmatch
import utils
import tensorflow as tf
from tqdm import tqdm
from tensorflow import keras


def find_files(root_dir, query="*.wav", include_root_dir=True):
    """Find files recursively.

    Args:
        root_dir (str): Root root_dir to find.
        query (str): Query to find.
        include_root_dir (bool): If False, root_dir name is not included.

    Returns:
        list: List of found filenames.

    """
    files = []
    for root, dirnames, filenames in os.walk(root_dir, followlinks=True):
        for filename in fnmatch.filter(filenames, query):
            files.append(os.path.join(root, filename))
    if not include_root_dir:
        files = [file_.replace(root_dir + "/", "") for file_ in files]

    return files

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate custom waveform files using pretrained InQSS-MOSnet.")
    parser.add_argument("--rootdir", default=None, type=str,
                        help="rootdir of the waveforms to be evaluated")
    parser.add_argument("--pretrained_model", type=str,
                        help="pretrained model file")
    parser.add_argument("--outputdir", type=str, default='Results',
                        help="outputdir of the results")
    args = parser.parse_args()

    #### tensorflow & gpu settings ####

    # 0 = all messages are logged (default behavior)
    # 1 = INFO messages are not printed
    # 2 = INFO and WARNING messages are not printed
    # 3 = INFO, WARNING, and ERROR messages are not printed
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

    tf.debugging.set_log_device_placement(False)
    # set memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    
    # find waveform files

    wavfiles = sorted(find_files(args.rootdir, "*.wav"))
    
    # init model
    print("Loading model")
    model = keras.models.load_model(args.pretrained_model)
    model_name = args.pretrained_model.split("/")[-2]
    print(model.summary())
    
    print("Start evaluating {} waveforms...".format(len(wavfiles)))
    results_qua = []
    results_intell = []

    output_dir = args.outputdir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for wavfile in tqdm(wavfiles):
        
        # get spectrogram
        mag_sgram = utils.get_spectrograms(wavfile)
        timestep = mag_sgram.shape[0]
        mag_sgram = np.reshape(mag_sgram,(1, timestep, utils.SGRAM_DIM))
        mag_sgram = (mag_sgram - mag_sgram.min()) / (mag_sgram.max() - mag_sgram.min())

        #get scattering coefficients
        _, scat1, scat2 = utils.extract_scatter(wavfile)
        scat = np.concatenate((scat1, scat2),axis=1)
        scat = (scat - scat.min()) / (scat.max() - scat.min())

        if scat.shape[0]!=timestep:
            tmp = np.zeros((timestep,54+179))
            tmp[:scat.shape[0],:] = scat
            scat = tmp

        scat = np.reshape(scat, (1,timestep, 54+179))

        # make prediction
        avg_qua, _, avg_intell, _ = model.predict([mag_sgram,scat], verbose=0, batch_size=1)

        # write to list
        result_qua = avg_qua[0][0]
        result_qua = wavfile + " {:.3f}".format(result_qua)
        results_qua.append(result_qua)
        
        result_intell = avg_intell[0][0]
        result_intell = wavfile + " {:.3f}".format(result_intell)
        results_intell.append(result_intell)

    # print average
    average_qua = np.mean(np.array([float(line.split(" ")[-1]) for line in results_qua]))
    average_intell = np.mean(np.array([float(line.split(" ")[-1]) for line in results_intell]))
    
    # write final raw result
    resultrawpath = os.path.join(output_dir, model_name+"_qua.txt")
    with open(resultrawpath, "w") as outfile:
        outfile.write("\n".join(sorted(results_qua)))
        outfile.write("\nAverage: {}\n".format(average_qua))

    
    resultrawpath = os.path.join(output_dir, model_name+"_intell.txt")
    with open(resultrawpath, "w") as outfile:
        outfile.write("\n".join(sorted(results_intell)))
        outfile.write("\nAverage: {}\n".format(average_intell))


if __name__ == '__main__':
    main()
