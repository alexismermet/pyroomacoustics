import numpy as np
from scipy.io import wavfile
import sounddevice as sd

import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import pyroomacoustics as pra
import matplotlib.pyplot as plt


"""
User parameters
"""
speech_file = "examples/input_samples/cmu_arctic_us_aew_a0001.wav"
noise_file = "examples/input_samples/doing_the_dishes.wav"
fact = 2
dest_noisy = "examples/output_samples/input_scnr.wav"
dest = "examples/output_samples/output_scnr.wav"

"""
Algo parameters
"""
db_reduc = 10
lookback = 10  # lookback this main samples for the noise floor estimate
fft_len = 512
beta = 30
alpha = 6.9

"""
Derived parameters and data
"""
n_fft_bins = fft_len//2 + 1
P_prev = np.zeros((n_fft_bins, lookback))
Gmin = 10**(-db_reduc/20)
G = np.zeros(n_fft_bins)

# make STFT object
hop = fft_len//2
window = pra.hann(fft_len, flag='asymmetric', length='full') 
stft = pra.realtime.STFT(fft_len, hop=hop, analysis_window=window, channels=1)

"""
Prepare input file
"""
fs_s, speech = wavfile.read(speech_file)
fs_n, noise = wavfile.read(noise_file)
max_val = abs(np.iinfo(speech.dtype).min)

# truncate to same length
noise = noise[:len(speech)]
actual_snr = 20*np.log10(np.linalg.norm(speech)/np.linalg.norm(fact*noise))
print("SNR [dB] = %f" % actual_snr)

noisy_signal = (speech + fact*noise).astype(np.float)
noisy_signal /= max_val
noisy_signal -= noisy_signal.mean()

# # play noisy file
# sd.default.samplerate = fs_s
# sd.play(noisy_signal, dtype='float64')
wavfile.write(dest_noisy, fs_s, noisy_signal.astype(np.float32))

"""
Process
"""
# collect the processed blocks
processed_audio = np.zeros(speech.shape)

n = 0
while noisy_signal.shape[0] - n > hop:

    # go to frequency domain
    stft.analysis(noisy_signal[n:(n+hop),])
    X = stft.X

    # estimate of signal + noise at current time
    P_sn = np.real(np.conj(X)*X)    

    # estimate of noise level
    P_prev[:,-1] = P_sn
    P_n = np.min(P_prev, axis=1)

    # compute mask
    for k in range(n_fft_bins):
        ##G[k] = max(((P_sn[k] - beta*P_n[k])/(P_sn[k]))**alpha, Gmin)
        G[k] = max((max(P_sn[k] - beta*P_n[k],0)/P_sn[k])**alpha, Gmin)

    # back to time domain
    processed_audio[n:n+hop,] = stft.synthesis(G*X)

    # update step
    P_prev = np.roll(P_prev, -1, axis=1)
    n += hop

"""
Write to wav
Plot spectrogram
"""
wavfile.write(dest,16000,processed_audio.astype(np.float32))

min_val = -80
max_val = -40
plt.figure()
plt.subplot(2,1,1)
plt.specgram(noisy_signal[:n-hop].astype(np.float32), NFFT=256, Fs=fs_s, vmin=min_val, vmax=max_val)
plt.title('Original Signal')
plt.subplot(2,1,2)
plt.specgram(processed_audio[hop:n].astype(np.float32), NFFT=256, Fs=fs_s, vmin=min_val, vmax=max_val)
plt.title('Filtered Signal')
plt.tight_layout(pad=0.5)
plt.show()