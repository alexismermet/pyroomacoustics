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
    dest_dir = "ouput_b(1)"
    labels_file = "conv_labels.txt"
    graph_file = "my_frozen_graph.pb"
    max_order = 3
    room_dim = [4,6]
    snr_vals = np.arange(100,-10,-10)
    desired_word = 'yes'
    pos_source = [1,4.5]
    pos_noise = [2.8,4.3]
    number_mics = 3
    mic = np.array([2,1.5])  #position
    d = 0.08                 #distance between microphones
    phi = 0.                 #angle from horizontal
    shape = 'Circular'       #array shape
    N = 1024                 #FFT length

    #Create the Microphone array
    if shape is 'Circular':
        R = pra.circular_2D_array(mic, number_mics, phi , d*number_mics/(2*np.pi)) 
    else:
        R = pra.linear_2D_array(mic, number_mics, phi, d)
    print(R)

    #create object
    dataset = pra.datasets.GoogleSpeechCommands(download=True,subset=1)
    print(dataset)

    #separate the noise and the speech samples
    noise_samps = dataset.filter(speech=0)
    speech_samps = dataset.filter(speech=1)
    speech_samps = speech_samps.filter(word=desired_word)

    #pick one of each from WAV
    speech = speech_samps[0]
    noise = noise_samps[1]

    print("speech file info :")
    print(speech.meta)
    print("noise file info:")
    print(noise.meta)
    print()

    #creating a noisy_signal array for each snr value
    speech_file_location = speech.meta.as_dict()['file_loc']
    noise_file_location = noise.meta.as_dict()['file_loc']

    #we use a signal going through our simple input_wav modifier defined in utils
    mics_array = np.array([[2.03819719,1.5],[1.98090141,1.53307973],[1.98090141,1.46692027]])
    noisy_signal_adder = utils.modify_input_wav_multiple_mics(speech_file_location,noise_file_location,room_dim,max_order,snr_vals,mics_array,pos_source,pos_noise)
    noisy_signal_beamformed = utils.modify_input_wav_beamforming(speech_file_location,noise_file_location,room_dim,max_order,snr_vals,R,pos_source,pos_noise,N)

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    score_adder = np.empty([len(snr_vals),number_mics])
    for i,snr in enumerate(snr_vals):
        for m in range(number_mics):
            dest = os.path.join(dest_dir,"snr_db_%d_mic_%d_adder.wav" %(snr,m))
            print(dest)
            noisy = (noisy_signal_adder[i][m]).astype(np.int16)
            wavfile.write(dest,16000,noisy)
            score_adder[i][m] = label_wav(dest, labels_file, graph_file, speech.meta.as_dict()['word'])

    score_adder_average = np.average(score_adder,axis=1)
    plt.plot(snr_vals,score_adder_average, label="not beamformed signal")
    plt.legend()

    #we use a signal going through beamforming

    score_beamformed = np.empty(len(snr_vals))
    for i,snr in enumerate(snr_vals):
        dest = os.path.join(dest_dir,"beamforming_signal_snr_db_%d.wav" %(snr))
        print(dest)
        noisy = pra.highpass(noisy_signal_beamformed[i],16000).astype(np.int16)
        wavfile.write(dest,16000,noisy)
        score_beamformed[i] = label_wav(dest, labels_file, graph_file, speech.meta.as_dict()['word'])

    plt.plot(snr_vals,score_beamformed, label="beamformed signal")
    plt.legend()
    plt.title('SNR agaisnt percentage of confidence')
    plt.show()
    


