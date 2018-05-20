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
    dest_dir = "ouput_full_multiple_words"
    labels_file = "conv_labels.txt"
    graph_file = "my_frozen_graph.pb"
    max_order = 3
    room_dim = [4,6]
    snr_vals = np.arange(50,-25,-10)
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
    desired_words = ['yes','no','up','down','left','right','on','off','stop','go']

    speech_samps = speech_samps.filter(word=desired_words)
    
    noise = noise_samps[0]
    noise_file_location = noise.meta.as_dict()['file_loc']
    speech_file_location = {} 
    for s in speech_samps:
        speech_file_location[s] = s.meta.as_dict()['file_loc'] 


    noisy_signal_beamformed = {}
    for s in speech_samps:
        noisy_signal_beamformed[s] = utils.modify_input_wav_beamforming(speech_file_location[s],noise_file_location,room_dim,max_order,snr_vals,R,pos_source,pos_noise,N) 

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    score_map = {}
    for w in desired_words:
    	score_map[w] = np.zeros([sub,len(snr_vals)])

    idx = 0
    for s in speech_samps:
   		for i,snr in enumerate(snr_vals):
   			word = s.meta.as_dict()['word']
   			dest = os.path.join(dest_dir,"beamforming_signal%d%s_snr_db_%d.wav" %(idx,word,snr))
   			noisy = noisy_signal_beamformed[s][i].astype(np.int16)
   			wavfile.write(dest,16000,noisy)
   			score_map[word][idx][i] = label_wav(dest, labels_file, graph_file, word)
   			idx +=1
   			if(idx == sub):
   				idx = 0
    score_map_average = {}			
    for w in desired_words:
      	score_map_average[w] = np.average(score_map[w], axis=0)

    plt.title('Classification of %s for %d given samples' %(desired_words,sub))
    plt.xlabel('SNR values [dB]')
    plt.ylabel('% confidence')
    for w in desired_words:
    	plt.plot(snr_vals,score_map_average[w],label=w)
    	plt.legend()
    plt.grid()
    plt.show()
   	
