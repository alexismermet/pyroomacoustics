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
    stft_in = pra.realtimes.STFT(fft_length, hop=hop, analysis_window=window, channels=1)
    
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