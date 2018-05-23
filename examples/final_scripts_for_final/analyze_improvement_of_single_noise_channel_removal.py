'''
Example of how to use single_noise_channel removal algorithm. In this example we also use what we have seen in other examples:
We're gonna synthetize a signal, Then we're gonna do processing on it using the algorithm and finally we are going to label
the newly obtained file and compare them to the file without any processing.
'''


import numpy as np
from scipy.io import wavfile
import pyroomacoustics.datasets.utils as utils

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

if __name__ == '__main__':

	'''
	User parameters for synthetizing the signal
	'''

	# how many order of reflection/refraction of the signal we consider in our rooms
	max_order = 3
	# the dimension of you room
	room_dim = [5,4,6]
	# the SNR values in dB we use to create the differents samples
	snr_vals = np.arange(60,-25,-5)
	# the number of mic you want placed in the room
	number_mics = 3
	# your microphones' array containing the position of your number_mics microphones you are going to use in the rooms
	mic_array = np.array([[2, 1.5, 2],[1,1,1],[1.5,2.5,4]])
	# desired basis word(s) (can also be a list)*
	desired_word = 'yes'
	#choose your label file
	labels_file = "conv_labels.txt"
    #choose your graph file
	graph_file = "my_frozen_graph.pb"
	# destination directory to write your new samples
	dest_dir = 'output_final_single_noise_removal'
	if not os.path.exists(dest_dir):
		os.makedirs(dest_dir)

	'''
	Parameters of the algortihm
	'''
	# the reduction criteria of the algorithm in dB
	db_reduc = 10
	# lookback this main samples for the noise floor estimate
	lookback = 10  
	# the lenght of our FFT
	fft_len = 512
	# value of the coefficient used to do the transition between the 2 values we are choosing from in the algorithm
	beta = 30
	alpha = 6.9
	# number of bins we're gonna use for our FFT
	n_fft_bins = fft_len//2 + 1 
	# array containing our powers that we use in our algorithm (we're gonna take the max value from this array)
	P_prev = np.zeros((n_fft_bins, lookback))
	# minmal value that our filter can take
	Gmin = 10**(-db_reduc/20)
	# our filter at the start: an empty array
	G = np.zeros(n_fft_bins)

	'''
	Selecting words from the dataset as in the example of the GoogleSpeechCommand
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
	noise = noise_samps[1]

	# print the information of our chosen speech and noise file
	print("speech file info :")
	print(speech.meta)
	print("noise file info:")
	print(noise.meta)
	print()

	'''
	Create new samples using Pyroomacoustics as in example how_to_synthesize_a_signal.py
	'''

	# creating a noisy_signal array for each snr value and mic
	speech_file_location = speech.meta.as_dict()['file_loc']
	noise_file_location = noise.meta.as_dict()['file_loc']
	noisy_signal = utils.modify_input_wav_multiple_mics(speech_file_location,noise_file_location,room_dim,max_order,snr_vals,mic_array,[2,3.1,2],[4,2,1.5])

	# reading our basis speech signal such that we can obtain its size
	fs_s, sound = wavfile.read(speech_file_location)

	# Create our new samples for each SNR values
	noisy = np.zeros([len(snr_vals),noisy_signal.shape[2]])
	for i,snr in enumerate(snr_vals):
		noisy[i] = np.average(noisy_signal[i],axis=0).astype('float32')

	'''
	make an STFT object (these class are already implemented in Pyroomacoustics and have example showing how to use them)
	'''

	hop = fft_len//2
	window = pra.hann(fft_len, flag='asymmetric', length='full') 
	stft = pra.realtime.STFT(fft_len, hop=hop, analysis_window=window, channels=1)

			
	'''
	Processing of our noisy signals contained in the noisy array.
	'''

	# collect the processed block for each of our noisy signal
	processed_audio_array = np.zeros([len(snr_vals),2*sound.shape[0]])

	# we run the algorithm for each of our possible signal
	for i,snr in enumerate(snr_vals):
		n = 0
		while noisy[i].shape[0] - n > hop:
			# go to frequency domain
			stft.analysis(noisy[i][n:(n+hop),])
			X = stft.X

    		# estimate of signal + noise at current time
			P_sn = np.real(np.conj(X)*X)    

    		# estimate of noise level
			P_prev[:,-1] = P_sn
			P_n = np.min(P_prev, axis=1)

    		# compute mask
			for k in range(n_fft_bins):
				G[k] = max((max(P_sn[k] - beta*P_n[k],0)/P_sn[k])**alpha, Gmin)

    		# back to time domain
			processed_audio_array[i][n:n+hop] = stft.synthesis(G*X)

    		# update step
			P_prev = np.roll(P_prev, -1, axis=1)
			n += hop

	'''
    Write to WAV + labelling of our processed noisy signals
    '''
    # labelling our different noisy signals
	score_processing = np.zeros(len(snr_vals))
	for i, snr in enumerate(snr_vals):
		dest = os.path.join(dest_dir,"single_noise_channel_signal_snr_db_%d.wav" %(snr))
		signal = processed_audio_array[i].astype(np.int16)
		wavfile.write(dest,16000,signal)
		score_processing[i] = label_wav(dest, labels_file, graph_file, speech.meta.as_dict()['word'])

    # plotting the result
	plt.plot(snr_vals,score_processing, label="single noise channel removal signal")
	plt.legend()
	plt.title('SNR agaisnt percentage of confidence')
	plt.xlabel('SNR in dB')
	plt.ylabel('score')
	plt.grid()
	plt.show()