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
    dest_dir = "ouput_full"
    labels_file = "conv_labels.txt"
    graph_file = "my_frozen_graph.pb"
    max_order = 3
    room_dim = [4,6]
    snr_vals = np.arange(100,-10,-10)
    pos_source = [1,4.5]
    pos_noise = [2.8,4.3]
    number_mics = 6
    mic = np.array([2,1.5])  #position
    d = 0.08                 #distance between microphones
    phi = 0.                 #angle from horizontal
    shape = 'Circular'       #array shape
    N = 1024                 #FFT length
    sub = 2

    #Create the Microphone array
    if shape is 'Circular':
        R = pra.circular_2D_array(mic, number_mics, phi , d*number_mics/(2*np.pi)) 
    else:
        R = pra.linear_2D_array(mic, number_mics, phi, d)
    R = np.concatenate((R, np.array(mic, ndmin=2).T), axis=1)

    #create object
    dataset = pra.datasets.GoogleSpeechCommands(download=True,subset=sub)
    print(dataset)

    #separate the noise and the speech samples
    noise_samps = dataset.filter(speech=0)
    speech_samps = dataset.filter(speech=1)
    desired_words = ['_silence_','_unknown_','yes','no','up','down','left','right','on','off','stop','go']
    print(desired_words)
    speech_samps = speech_samps.filter(word=desired_words)

    print(speech_samps)
    
    noise = noise_samps[0]
    noise_file_location = noise.meta.as_dict()['file_loc']
    speech_file_location = {} 
    for s in speech_samps:
        speech_file_location[s] = s.meta.as_dict()['file_loc'] 


    noisy_signal_beamformed = {}
    for s in speech_samps:
        noisy_signal_beamformed[s] = utils.modify_input_wav_beamforming(speech_file_location[s],noise_file_location,room_dim,max_order,snr_vals,R,pos_source,pos_noise,N) 
    print(noisy_signal_beamformed)

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    score = {'_silence_':np.array([sub,len(snr_vals)]),'_unknown_':np.array([sub,len(snr_vals)]),'yes':np.array([sub,len(snr_vals)]),'no':np.array([sub,len(snr_vals)]),'up':np.array([sub,len(snr_vals)]),'down':np.array([sub,len(snr_vals)]),'left':np.array([sub,len(snr_vals)]),'right':np.array([sub,len(snr_vals)]),'on':np.array([sub,len(snr_vals)]),'off':np.array([sub,len(snr_vals)]),'stop':np.array([sub,len(snr_vals)]),'go':np.array([sub,len(snr_vals)])}
    index = 0
    for s in speech_samps:
        curr = s.meta.as_dict()['word']
        for i,snr in enumerate(snr_vals):
            dest = os.path.join(dest_dir,"beamforming_signal%d_snr_db_%d.wav" %(index,snr))
            noisy = noisy_signal_beamformed[s][i].astype(np.int16)
            wavfile.write(dest,16000,noisy)
            score[s.meta.as_dict()['word']][index][i] = label_wav(dest, labels_file, graph_file, s.meta.as_dict()['word'])
        pred = curr
        index += 1
        if(pred != curr):
            index = 0

    print(score)