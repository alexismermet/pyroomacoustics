'''
Google Speech Commands Dataset
=================
Pyroomacoustics includes a wrapper around the Google Speech Commands dataset [TODO add reference].
'''



import sys
import numpy as np
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import pyroomacoustics as pra
import os, argparse
import pyroomacoustics.datasets.utils as utils
import matplotlib.pyplot as plt
from scipy.io import wavfile 

# import tf and functions for labelling
import tensorflow as tf

from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

def  load_graph(f):
    with tf.gfile.FastGFile(f,'rb') as graph:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(graph.read())
        tf.import_graph_def(graph_def, name='')


def load_labels(f):
    return [line.rstrip() for line in tf.gfile.GFile(f)]


def run_graph(wav_data, labels, index, how_many_labels=3):
    with tf.Session() as session:
        softmax_tensor = session.graph.get_tensor_by_name("labels_softmax:0")
        predictions, = session.run(softmax_tensor,{"wav_data:0": wav_data})

    top_k = predictions.argsort()[-how_many_labels:][::-1]
    for node_id in top_k:
        human_string = labels[node_id]
        score = predictions[node_id]
        print('%s (score = %.5f)' % (human_string, score))
    return predictions[index]



def label_wav(wav,labels,graph,word):

    if not wav or not tf.gfile.Exists(wav):
        tf.logging.fatal('Audio file does not exist %s',wav)
    if not labels or not tf.gfile.Exists(labels):
        tf.logging.fatal('Labels file does not exist %s', labels)
    if not graph or not tf.gfile.Exists(graph):
        tf.logging.fatal('Graph file does not exist %s', graph)

    labels_list = load_labels(labels)
    load_graph(graph)

    with open(wav,'rb') as wav_file:
        wav_data = wav_file.read()
    index = labels_list.index(word)
    return run_graph(wav_data,labels_list,index)



if __name__ == '__main__':

    # user parameters
    dest_dir = "output_noise_supp"
    labels_file = "conv_labels.txt"
    graph_file = "my_frozen_graph.pb"
    max_order = 3
    room_dim = [5,4,6]
    desired_word = 'yes'
    pos_source = np.array([2.5,2.5,2.5])
    pos_noise = np.array([3,2,1.5])
    mic_array = np.array([[2, 1.5, 2],[1,1,1],[1.5,2.5,4]])
    fft_length = 500
    alpha = 1.2
    beta = 0.5

    # create object
    dataset = pra.datasets.GoogleSpeechCommands(download=True,subset=1)
    print(dataset)

    # separate the noise and speech samples
    noise_samps = dataset.filter(speech=0)
    speech_samps = dataset.filter(speech=1)

    desired_words = np.array(['_silence_','_unknown_','yes','no','up','down','left','right','on','off','stop','go'])
    print(desired_words)

    speech_samps = speech_samps.filter(word=desired_word)

    # pick one of each from WAV
    speech = speech_samps[0]
    noise = noise_samps[1]

    print("speech file info :")
    print(speech.meta)
    print("noise file info:")
    print(noise.meta)
    print()

    speech_file_location = speech.meta.file_loc
    noise_file_location = noise.meta.file_loc

    noisy_signal = pra.noise_supp.noise_suppressor(speech_file_location,noise_file_location,room_dim,pos_source,pos_noise,max_order,mic_array,fft_length,alpha,beta)

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    dest = os.path.join(dest_dir,"noisy_signal.wav")
    wavfile.write(dest,16000,noisy_signal)
    score = label_wav(dest, labels_file, graph_file, speech.meta.as_dict()['word'])
    print(score)