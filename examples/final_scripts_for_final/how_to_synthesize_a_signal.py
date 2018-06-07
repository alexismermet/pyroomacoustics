'''
Example of how to synthesize a sound in pyroomacoustics using the function modify_input_wav_multiple_mics from datasets/utils.py
In this example we are just gonna synthetize new signals from a given one and with a given noise for a given SNR vals.
Afterwards you can use these new created samples as in the example how_to_label_a_file.py 

In this example we're gonna use the GoogleSpeechDataset to select the speech sound and also the noise
We're going to do as in the example already in pyroomacoustics : google_speech_commands_corpus.py

'''

import sys
import numpy as np
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import pyroomacoustics as pra
import os, argparse
import utils
import matplotlib.pyplot as plt
from scipy.io import wavfile 

# import sounddevice if you want to listen the sound you have created (you can comment this part if you dont want to).
import sounddevice as sd


if __name__ == '__main__':
	# how many order of reflection/refraction of the signal we consider in our rooms
	max_order = 3
	# the dimension of you room
	room_dim = [5,4,6]
	# the SNR value in dB we use to create the new sample (here you can give a list if you
	# want to create multiples samples. The code is already ready for this change)
	snr_vals = [20]
	# the number of mic you want placed in the room
	number_mics = 3
	# your microphones' array containing the position of your number_mics microphones you are going to use in the rooms
	mic_array = np.array([[2, 1.5, 2],[1,1,1],[1.5,2.5,4]])
	# your input wav you want to modify
	speech_file_location = 'examples/final_scripts_for_final/00b01445_nohash_0.wav'
	# your noise
	noise_file_location = 'examples/final_scripts_for_final/pink_noise.wav'

	# enable the sounddevice part
	use_sounddevice = False
	# destination directory to write your new samples
	dest_dir = 'output_final_synthesis'
	if not os.path.exists(dest_dir):
		os.makedirs(dest_dir)

	'''
	Create our new samples using pyroomacoustics
	'''

	# creating a noisy_signal array for each snr value
	noisy_signal = utils.modify_input_wav_multiple_mics(speech_file_location,noise_file_location,room_dim,max_order,snr_vals,mic_array,[2,3.1,2],[4,2,1.5])

	# listen to your new sample or just write them to wav
	for i,snr in enumerate(snr_vals):
			noisy = np.average(noisy_signal[i],axis=0).astype('float32')
			dest = os.path.join(dest_dir,"snr_db_%d.wav" %(snr))
			if(use_sounddevice):
				sd.play(noisy, samplerate = 16000)
				sd.wait()
			wavfile.write(dest,16000,noisy)

	'''
	 plot spectograms
	'''
	#recover the orignal sample
	fs_s, audio_anechoic = wavfile.read(speech_file_location)

	min_val = -80
	max_val = 0
	plt.figure()

	# plot the original signal
	plt.specgram(audio_anechoic/np.linalg.norm(audio_anechoic).astype('float32'), NFFT=256, Fs=fs_s, vmin=min_val, vmax=max_val)
	plt.title('Original Speech signal')
	
	# plot the new signals
	for i,snr in enumerate(snr_vals):
		noisy = np.average(noisy_signal[i],axis=0).astype('float32')
		plt.figure()
		plt.specgram(noisy, NFFT=256, Fs=fs_s, vmin=min_val, vmax=max_val)
		plt.title('new samples at snr values %d' %(snr))

	plt.tight_layout(pad=0.5)
	plt.show()