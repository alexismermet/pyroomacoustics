'''
Example that shows the improvement of using beamforming on a signal and also how tu use beamforming followed by labelling on pyroomacoustics
'''

import numpy as np
from scipy.io import wavfile
import utils

import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import pyroomacoustics as pra
import matplotlib.pyplot as plt

# import tf and functions for labelling
import tensorflow as tf


from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

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

if __name__ =='__main__':

	'''
	User parameters for synthetizing the signal
	'''

	# how many ordeer of reflection\refraction of the signal we consider in our rooms
	max_order = 3
	# the dimension of your room
	room_dim = [4,6] 
	# the SNR values in dB we use to create the different samples
	snr_vals = np.arange(60,-25,-5)
	#position of the sound source
	pos_source = [1,4.5]
	#position of the noise source
	pos_noise = [2.8,4.3]
	#the number of mic you want to place in the room
	number_mics = 3
	
	# creation of the mic_array in a special way such that we can use beamforming
	# shape of the array
	shape = 'Circular'
	#position of the center mic
	mic = np.array([2,1.5]) 
	# radius of the array
	d = 0.2
	# the angle from horizontal
	phi = 0. 
	# creation of the array
	if shape is 'Circular':
		R = pra.circular_2D_array(mic, number_mics, phi , radius=d) 
	else:
		R = pra.linear_2D_array(mic, number_mics, phi, d)
	R = np.concatenate((R, np.array(mic, ndmin=2).T), axis=1)
	# FFT length
	N = 1024
	
	# desired basis word(s) (can also be a list)
	desired_word = 'yes'
	#choose your label file
	labels_file = "conv_labels.txt"
	# choose your graph file
	graph_file = "my_frozen_graph.pb"
	# destination directory to write your new samples
	dest_dir = 'output_final_beamforming'
	if not os.path.exists(dest_dir):
		os.makedirs(dest_dir)

	'''
	Selecting words from the dataset as in the example of the GoogleSpeechComman
	'''
	# create the dataset object
	dataset = pra.datasets.GoogleSpeechCommands(download=True,subset=1)

	# separate the noise and the speech samples
	noise_samps = dataset.filter(speech=0)
	speech_samps = dataset.filter(speech=1)
	# filter the speech samples to take only the desired word(s)
	speech_samps = speech_samps.filter(word=desired_word)

	# pick one sample of each (from the noise samples and the speech samples filtered)
	speech = speech_samps[0]
	noise = noise_samps[0]

	# print the information of our chosen speech and noise file
	print("speech file info :")
	print(speech.meta)
	print("noise file info:")
	print(noise.meta)
	print()

	'''
	create new samples using Pyroomacoustics
	'''
	# creating a noisy_signal array for each snr value
	speech_file_location = speech.meta.as_dict()['file_loc']
	noise_file_location = noise.meta.as_dict()['file_loc']
	# we're gonna work with only the central mic to create our noisy signal.
	noisy_signal= utils.modify_input_wav_multiple_mics(speech_file_location,noise_file_location,room_dim,max_order,snr_vals,np.array([mic]),pos_source,pos_noise)
	# we create our  beamformed noisy_signal which directly apply the algorithm to it using our function in utils.py
	noisy_signal_beamformed = utils.modify_input_wav_beamforming(speech_file_location,noise_file_location,room_dim,max_order,snr_vals,R,pos_source,pos_noise,N)

	'''
	Write to WAV + labelling of our processed noisy signals
	'''
	# we flatten by one dimension our array. In fact we just say that it does'nt have a microphones dimension
	noisy_signal_flatten = noisy_signal[:,0,:]

	 # labelling our beamformed signals and comparing their classification with the one for the original noisy signals
	score_processing = np.zeros(len(snr_vals))
	score_original = np.zeros(len(snr_vals))

	for i, snr in enumerate(snr_vals):
		print("SNR / %f dB" %snr)
		dest = os.path.join(dest_dir, "beamformed_signal_snr_db_%d.wav" %snr)
		signal = noisy_signal_beamformed[i].astype(np.int16)
		wavfile.write(dest,16000,signal)
		score_processing[i] = label_wav(dest, labels_file, graph_file, speech.meta.as_dict()['word'])

		dest = os.path.join(dest_dir,"original_signal_snr_db_%d.wav" %(snr))
		signal = noisy_signal_flatten[i].astype(np.int16)
		wavfile.write(dest,16000,signal)
		score_original[i] = label_wav(dest, labels_file, graph_file, speech.meta.as_dict()['word'])
		print()

	#plotting the result
	plt.plot(snr_vals,score_processing, label="beamformed signal")
	plt.plot(snr_vals,score_original, label="original")
	plt.legend()
	plt.title('SNR against percentage of confidence')
	plt.xlabel('SNR in dB')
	plt.ylabel('score')
	plt.grid()
	plt.show()	