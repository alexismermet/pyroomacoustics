#The Tenserflow Authors
#2017
#label_wav.py (Version 2.0) [Source Code].
#https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/speech_commands/label_wav.py
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from scipy.io import wavfile

import argparse
import sys

import tensorflow as tf

from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

FLAGS = None

"""
Example of how to run:

python augmenting_data.py --wav ffd2ba2f_nohash_4.wav --noise white_noise.wav --dest_wav output --labels conv_labels.txt --graph my_frozen_graph.pb --room_dim 4 5 6

"""

def modify_input_wav(wav,noise,room_dim,max_order,snr_vals):

	fs_s, audio_anechoic = wavfile.read(wav)
	fs_n, noise_anechoic = wavfile.read(noise)
	
	#Create a room for the signal
	room_signal= pra.ShoeBox(
		room_dim,
		absorption = 0.2,
		fs = fs_s,
		max_order = max_order)

	#rCeate a room for the noise
	room_noise = pra.ShoeBox(
		room_dim,
		absorption=0.2,
		fs=fs_n,
		max_order = max_order + 1)

	#source of the signal and of the noise in their respectiv boxes
	room_signal.add_source([2,3.1,2],signal=audio_anechoic)
	room_noise.add_source([4,2,1.5], signal=noise_anechoic)

	#we add a microphone at the same position in both of the boxes
	room_signal.add_microphone_array(
		pra.MicrophoneArray(
	        np.array([[2, 1.5, 2]]).T, 
	        room_signal.fs)
	    )
	room_noise.add_microphone_array(
		pra.MicrophoneArray(
	        np.array([[2, 1.5, 2]]).T, 
	        room_noise.fs)
	    )

	#simulate both rooms
	room_signal.simulate()
	room_noise.simulate()

	#take the mic_array.signals from each room
	audio_reverb = room_signal.mic_array.signals
	noise_reverb = room_noise.mic_array.signals

	#verify the size of the two arrays such that we can continue working on the signal
	if(len(noise_reverb) < len(audio_reverb)):
		raise ValueError('the length of the noise signal is inferior to the one of the audio signal !!')

	#normalize the noise
	print(np.shape(audio_reverb))
	noise_reverb = noise_reverb[:,:np.shape(audio_reverb)[1]]
	print(np.shape(noise_reverb))
	noise_normalized = noise_reverb/np.linalg.norm(noise_reverb)

	noisy_signal = {}

	for snr in snr_vals:
		noise_std = np.linalg.norm(audio_reverb)/(10**(snr/20.))
		final_noise = noise_normalized*noise_std
		noisy_signal[snr] = audio_reverb + final_noise
	return noisy_signal


	"""
	1) new input to function (array of SNR values) --> `snr_vals` (you can create it with np.arange)
	2) simulate signal in room --> room_signal.mic_array.signals
	3) simulate noise in room --> room_noise.mic_array.signals
	4) MAKE SURE THAT LENGTH OF `room_noise.mic_array.signals` IS GREATER THAN
	   OR EQUAL TO LENGTH OF `room_signal.mic_array.signals`
	   - this can be probably be done by setting the `max_order` of the room 
	   	 simulating the noise larger than the `max_order` of the room 
	   	 simulating the signal/speech
	5) truncate `room_noise.mic_array.signals` so that it is same length as
       `room_signal.mic_array.signals`
    6) Normalize noise:
		for each microphone m:
		 	noise_normalized[m,:] = room_noise.mic_array.signals[m,:] / np.linalg.norm(room_noise.mic_array.signals[m,:])
    6) for each snr in snr_vals:
		 initialize array for `noisy signal`
		 for each microphone m:
		 	noise_std = np.linalg.norm(room_signal.mic_array.signals[m,:]) / (10**(snr/20.))
		 	noise = noise_normalized[m,:]*noise_std
		 	noisy_signal[m,:] = room_signal.mic_array.signals[m,:] + noise
		 save `noisy_signal` as 'reverb_output_snr_level'
	7) Step 6 should create a WAV file (using scipy.io.wavfile.write) for each 
	   snr level in `snr_vals`. Then you can compute the classification value 
	   for each of those files.
    8) plot `snr_vals` against classification value

	"""


def  load_graph(f):
	with tf.gfile.FastGFile(f,'rb') as graph:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(graph.read())
		tf.import_graph_def(graph_def, name='')


def load_labels(f):
	return [line.rstrip() for line in tf.gfile.GFile(f)]


def run_graph(wav_data, labels, how_many_labels):
	with tf.Session() as session:
		softmax_tensor = session.graph.get_tensor_by_name("labels_softmax:0")
		predictions, = session.run(softmax_tensor,{"wav_data:0": wav_data})

	top_k = predictions.argsort()[-how_many_labels:][::-1]
	for node_id in top_k:
		human_string = labels[node_id]
		score = predictions[node_id]
		print('%s (score = %.5f)' % (human_string, score))
	return score


def label_wav(wav,labels,graph,how_many_labels):

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

	return run_graph(wav_data,labels_list,how_many_labels)


def main(_):
	snr_vals = np.arange(100,-10,-10)
	fs, x = wavfile.read(FLAGS.wav)
	
	noisy_signal = modify_input_wav(FLAGS.wav,FLAGS.noise,FLAGS.room_dim,FLAGS.max_order,snr_vals)
	
	i = 0
	correctness = np.empty(len(snr_vals))
	for snr in snr_vals:
		dest = FLAGS.dest_wav + str(snr) + '.wav' 
		noisy =noisy_signal[snr]
		print(noisy)
		wavfile.write(dest,fs,noisy[0])
		correctness[i] = label_wav(dest, FLAGS.labels, FLAGS.graph, FLAGS.how_many_labels)

	plt.plot(snr_vals,correctness)
	plt.title('SNR against percentage of confidence')
	plt.show()




if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--wav', type=str, default='', help='the audio file you want processed and then identified.')
	parser.add_argument(
		'--noise', type=str, default='', help='the noise you want to be added to your audio file to be processed')
	parser.add_argument(
		'--graph', type=str, default='', help='the model you want to use for identification.')
	parser.add_argument(
		'--labels', type=str, default='', help='the path to the file containing the labels for your data.')
	parser.add_argument(
		'--dest_wav', type=str, default='', help='the place where you want the processed data to be saved before using it with the model.')
	parser.add_argument('--room_dim', nargs='+', type=int, default=[5,4,6], help='give the different coordinates for a 3D shoebox room.')
	parser.add_argument(
		'--max_order', type=int, default=3, help='the number of reflection you want to do.')
	parser.add_argument(
		'--how_many_labels', type=int, default=3, help='Number of result to show.')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)