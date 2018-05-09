import os, tarfile, bz2, requests, gzip 
from scipy.io import wavfile
import numpy as np
import pyroomacoustics as pra
from scipy.signal import fftconvolve


def noise_suppressor(wav,noise,room_dim,pos_source,pos_noise,max_order,mic_pos,fft_length,alpha,beta):

	fs_s, audio_anechoic = wavfile.read(wav)
    fs_n, noise_anechoic = wavfile.read(noise)

	room= pra.ShoeBox(
    	room_dim,
    	absorption = 0.2,
    	fs = fs_s,
        max_order = max_order)

	room.add_source(pos_source,signal=audio_anechoic)
    room.add_source(pos_noise, signal=noise_anechoic)

    room.add_microphone_array(
    pra.MicrophoneArray(
        mic_pos.T, 
        room_signal.fs)
    )

    room.simulate()
    x = room.mic_array.signals

    #we create the STFT of our noisy signal
    hop = fft_length//2
    window = pra.hann(fft_length)
    stft_in = pra.realtime.STFT(fft_length, hop=hop, analysis_window=window, channels=1)
    
    stft_in.analysis(x)
    X = stft_in.X

    #creation of the filter
    power = np.square(np.absolute(X))
    p_Noise = np.amin(power)	

    Gmin = 10**(-10/20)
    G = np.zeroes(len(X))

    for k in range(len(X)):
    	G[k] = max((power[k]-beta*p_Noise**alpha)/(power[k]**alpha),Gmin)

    stft_in.set_filter(G,freq=True)
    stft_in.process()

    output = stft_in.synthesis()
    return pra.normalize(output, bits=16)

def noise_suppressor2(wav,noise,room_dim,pos_source,pos_noise,max_order,mic_pos,fft_length,alpha,beta,snr_vals):

	fs_s, audio_anechoic = wavfile.read(wav)
    fs_n, noise_anechoic = wavfile.read(noise)

	room_audio = pra.ShoeBox(
    	room_dim,
    	absorption = 0.2,
    	fs = fs_s,
        max_order = max_order)

	room_noise = pra.ShoeBox(
    	room_dim,
    	absorption = 0.2,
    	fs = fs_s,
        max_order = max_order)

	room_audio.add_source(pos_source,signal=audio_anechoic)
    room_noise.add_source(pos_noise, signal=noise_anechoic)

    room_audio.add_microphone_array(
    pra.MicrophoneArray(
        mic_pos.T, 
        room_signal.fs)
    )

    room_noise.add_microphone_array(
    pra.MicrophoneArray(
        mic_pos.T, 
        room_signal.fs)
    )

    room_audio.simulate()
    room_noise.simulate()

    #take the mic_array.signals from each room
    x = room_audio.mic_array.signals
    n = room_noise.mic_array.signals

    shape = np.shape(x)

    noise_normalized = np.zeros(shape)

    #for each microphones
    if(len(n[0]) < len(x[0])):
        raise ValueError('the length of the noise signal is inferior to the one of the audio signal !!')
    n = n[:,:len(x[0])]

    norm_fact = np.linalg.norm(n[0])
    noise_normalized = n / norm_fact

    #initilialize the array of noisy_signal
    noisy_signal = np.zeros([len(snr_vals),shape[1]])

    for i,snr in enumerate(snr_vals):
        noise_std = np.linalg.norm(x[0])/(10**(snr/20.))
        for m in range(shape[0]):
            
            final_noise = noise_normalized[m]*noise_std
            noisy_signal[i] += x[m] + final_noise
        noisy_signal[i] = noisy_signal[i]/shape[0]

    final_signal = np.zeros(len(snr_vals))
    #we create the STFT of our noisy signal
    hop = fft_length//2
    window = pra.hann(fft_length)
    
    for i,xn in enumerate(noisy_signal):
    	stft_in = pra.realtimes.STFT(fft_length, hop=hop, analysis_window=window, channels=1)
    
    	stft_in.analysis(xn)
    	XN = stft_in.X

	    #creation of the filter
	    power = np.square(np.absolute(X))
	    p_Noise = np.amin(power)	

	    Gmin = 10**(-10/20)
	    G = np.zeroes(len(X))

	    for k in range(len(X)):
	    	G[k] = max((power[k]-beta*p_Noise**alpha)/(power[k]**alpha),Gmin)

	    stft_in.set_filter(G,freq=True)
	    stft_in.process()

	    output = stft_in.synthesis()
	    final_signal[i] = output

	return final_signal
