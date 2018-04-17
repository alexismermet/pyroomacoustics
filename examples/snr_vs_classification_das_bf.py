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
# import pyroomacoustics.datasets.utils as utils
import matplotlib.pyplot as plt
from scipy.io import wavfile 

# import tf and functions for labelling
import tensorflow as tf

from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

def load_graph(f):
    with tf.gfile.FastGFile(f,'rb') as graph:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(graph.read())
        tf.import_graph_def(graph_def, name='')


def load_labels(f):
    return [line.rstrip() for line in tf.gfile.GFile(f)]


def run_graph(wav_data, labels, index, how_many_labels=3, verbose=False):
    with tf.Session() as session:
        softmax_tensor = session.graph.get_tensor_by_name("labels_softmax:0")
        predictions, = session.run(softmax_tensor,{"wav_data:0": wav_data})

    top_k = predictions.argsort()[-how_many_labels:][::-1]
    for node_id in top_k:
        human_string = labels[node_id]
        score = predictions[node_id]
        if verbose:
            print('%s (score = %.5f)' % (human_string, score))
    return predictions[index]



def label_wav(wav,labels,graph,word):
    """
    Requires WAV file to be coded as int16
    """

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

    dest_dir = "bf_output"
    labels_file = "conv_labels.txt"
    graph_file = "my_frozen_graph.pb"
    max_order = 3
    absorption_fact = 0.2
    room_dim = [4,6]
    snr_vals = np.arange(20,-30,-5)
    desired_word = 'yes'
    pos_source = [1,4.5]
    pos_noise = [2.8,4.3]
    fft_len = 1024

    # use circular array with center mic
    center = np.array([2,1.5])
    radius = 0.2
    R = pra.circular_2D_array(center, M=6, phi0=0, radius=radius)
    R = np.concatenate((R, np.array(center, ndmin=2).T), axis=1)

    # visualize the setup
    room = pra.ShoeBox(room_dim, absorption=absorption_fact, max_order=max_order)
    room.add_source(pos_source)
    room.add_source(pos_noise)
    room.add_microphone_array(pra.Beamformer(R, room.fs, N=fft_len))
    room.mic_array.rake_delay_and_sum_weights(room.sources[0][:1])
    room.plot(freq=[500, 1000, 2000, 4000], img_order=0)
    plt.title("Simulation setup and polar patterns")
    plt.legend(['500', '1000', '2000', '4000'])
    plt.grid()

    #create object
    dataset = pra.datasets.GoogleSpeechCommands(download=True,subset=1)

    #separate the noise and the speech samples
    noise_samps = dataset.filter(speech=0)
    speech_samps = dataset.filter(speech=1)
    speech_samps = speech_samps.filter(word=desired_word)

    #pick one of each from WAV
    speech_samp = speech_samps[0]
    noise_samp = noise_samps[0]
    print()
    print("SPEECH FILE INFO :")
    print(speech_samp)
    print("NOISE FILE INFO :")
    print(noise_samp)
    print()

    #creating a noisy_signal array for each snr value
    speech_file_location = speech_samp.meta.file_loc
    noise_file_location = noise_samp.meta.file_loc

    """
    Beamform original signal.
    
    First the room with only the signal.
    """
    fs_s, speech = wavfile.read(speech_file_location)
    input_type = speech.dtype
    try:
        IN_MAX_VAL = max(np.iinfo(input_type).max, abs(np.iinfo(input_type).min))
    except:
        IN_MAX_VAL = max(np.finfo(input_type).max, abs(np.finfo(input_type).min))

    room_sig = pra.ShoeBox(room_dim, absorption=absorption_fact, fs=fs_s, 
        max_order=max_order)
    room_sig.add_source(pos_source, signal=speech)
    room_sig.add_microphone_array(pra.Beamformer(R, fs_s, N=fft_len))
    room_sig.simulate()
    room_sig.mic_array.rake_delay_and_sum_weights(room_sig.sources[0][:1])
    speech_bf = room_sig.mic_array.process()

    room_sig.plot(freq=[500, 1000, 2000, 4000], img_order=0)
    plt.title("Room (signal)")
    plt.legend(['500', '1000', '2000', '4000'])
    plt.grid()

    """
    Now the room with just the noise. MUST BE SAME BEAMFORMER
    """
    fs_n, noise = wavfile.read(noise_file_location)
    if fs_s != fs_n:
        raise ValueError("Sampling frequencies not equal!")
    room_noise = pra.ShoeBox(room_dim, absorption=absorption_fact, fs=fs_n, 
        max_order=max_order)
    room_noise.add_source(pos_noise, signal=noise)
    room_noise.add_microphone_array(pra.Beamformer(R, fs_n, N=fft_len))
    room_noise.simulate()
    room_noise.mic_array.rake_delay_and_sum_weights(room_sig.sources[0][:1])
    noise_bf = room_noise.mic_array.process()

    room_noise.plot(freq=[500, 1000, 2000, 4000], img_order=0)
    plt.title("Room (noise)")
    plt.legend(['500', '1000', '2000', '4000'])
    plt.grid()

    """
    We wish to compute the SNR wrt to same (single) microphone. Let's use the
    center microphone.
    """
    ref_mic_sig = room_sig.mic_array.signals[-1,:]
    ref_mic_noise = room_noise.mic_array.signals[-1,:]

    # truncate noise to same length
    ref_mic_noise = ref_mic_noise[:len(ref_mic_sig)]

    # norm factor for signal and noise
    sig_lvl = np.linalg.norm(ref_mic_sig)
    noise_norm_fact = np.linalg.norm(ref_mic_noise)
    snr_facts = sig_lvl * 10**(-snr_vals/20) / noise_norm_fact  # multiple noise by this

    # make sure factors computed correctly
    print()
    print("CHECKING THAT SNR FACTORS ARE COMPUTED CORRECTLY")
    for idx, fact in enumerate(snr_facts):
        signal_lvl = np.linalg.norm(ref_mic_sig)
        noise_lvl = np.linalg.norm(ref_mic_noise * fact)
        print("Expected SNR : %f" % snr_vals[idx])
        snr_db = 20*np.log10(signal_lvl/noise_lvl)
        print("SNR : %f" % snr_db)

    """
    Weight and add beamformed output from signal and noise to simulate beamforming
    under different SNRs.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # truncate beamformed noise
    noise_bf = noise_bf[:len(speech_bf)]

    # compute score for different SNR vals
    print()
    score_beamformed = np.empty(len(snr_vals))
    score_single = np.empty(len(snr_vals))
    for idx, snr in enumerate(snr_vals):

        noisy_signal = speech_bf + snr_facts[idx]*noise_bf
        noisy_signal = pra.normalize(pra.highpass(noisy_signal, fs_s), bits=16).astype(np.int16)
        dest = os.path.join(dest_dir,"das_bf_snr_db_%d.wav" %(snr))
        wavfile.write(dest, fs_s, noisy_signal)
        score_beamformed[idx] = label_wav(dest, labels_file, graph_file, 
            speech_samp.meta.word)

        # compute score for single mic for reference
        single_mic = ref_mic_sig + snr_facts[idx]*ref_mic_noise
        single_mic = pra.normalize(pra.highpass(single_mic, fs_s), bits=16).astype(np.int16)
        dest = os.path.join(dest_dir,"single_mic_snr_db_%d.wav" %(snr))
        wavfile.write(dest, fs_s, single_mic)
        score_single[idx] = label_wav(dest, labels_file, graph_file, 
            speech_samp.meta.word)


    plt.figure()
    plt.plot(snr_vals,score_beamformed, label="beamformed signal")
    plt.plot(snr_vals,score_single, label="single mic (center)")
    plt.legend()
    plt.grid()
    plt.ylabel("Score")
    plt.xlabel("SNR [dB]")
    plt.title('Classification for : ' + speech_samp.meta.word)

    plt.show()
    


