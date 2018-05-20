'''
Example of how to label a file using the GoogleSpeechCommand Dataset and the graph obtained from the Tensorflow example
For this example you need to have on your computer (or to install if you haven't):
	-Tensorflow
	-sounddevice (if you want to record your own sound)
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

# import sounddevice if you want to use it (you can comment this part if you dont want to).
import sounddevice as sd


# here we recreate some function from tensorflow to be able to extract the results obtained via labelling and to plot them if we want to.

# load the graph we're gonna use for labelling
def  load_graph(f):
    with tf.gfile.FastGFile(f,'rb') as graph:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(graph.read())
        tf.import_graph_def(graph_def, name='')

# load the labels we're gonna use with the graph
def load_labels(f):
    return [line.rstrip() for line in tf.gfile.GFile(f)]

# run the graph and label our file. We add the fact that this function returns the prediction such that we can work with it afterwards.
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


# main function used for labelling. We add a retrun to this function to recover the results.
# this function labels wavfiles so you always need to create a wavfile of your sound to label it.

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
	#choose the directory where you're gonna create the wavfile.
    dest_dir = "ouput_final"
    #choose your wavfile name (the one you want to load or the name of your record. Dont forget the .wav)
    wav = 'test.wav'
    #choose your label file
    labels_file = "conv_labels.txt"
    #choose your graph file
    graph_file = "my_frozen_graph.pb"
    #the word you want labelled. It should be for our graph in the set ['yes','no','up','down','left','right','on','off','stop','go']
    word = 'no'
    #set to True if you want to record your own sound and set the other parameters as you wish
    use_sounddevice = True
    time_recording = 1
    sample_rate = 44100

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    # create the wavfile to label or just load it
    if(use_sounddevice):
    	#record your voice 
    	print('recording\n')
    	sound= sd.rec(int(time_recording * sample_rate), samplerate= sample_rate, channels= 1, dtype='int16')
    	sd.wait()
    	print('done recording\n')
    	#replay the recording
    	sd.play(sound)
    	sd.wait
    	#write your wavfile
    	destination = os.path.join(dest_dir, wav)
    	wavfile.write(destination,44100,sound)
    else:
    	destinatinon = wav
    
    #label your wavfile
    score = label_wav(destination, labels_file, graph_file, word)
    print('the score obtained for this recording for the word %s is:\n' %word)
    print('%d%%' %(score*100))


