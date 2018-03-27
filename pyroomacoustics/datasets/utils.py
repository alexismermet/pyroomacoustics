import os, tarfile, bz2, requests, gzip 
from scipy.io import wavfile
import numpy as np
import pyroomacoustics as pra

def download_uncompress_tar_bz2(url, path='.'):

    # open the stream
    r = requests.get(url, stream=True)

    tmp_file = 'temp_file.tar'

    # Download and uncompress at the same time.
    chunk_size = 4 * 1024 * 1024  # wait for chunks of 4MB
    with open(tmp_file, 'wb') as file:
        decompress = bz2.BZ2Decompressor()
        for chunk in r.iter_content(chunk_size=chunk_size):
            file.write(decompress.decompress(chunk))

    # finally untar the file to final destination
    tf = tarfile.open(tmp_file)
    tf.extractall(path)

    # remove the temporary file
    os.unlink(tmp_file)


def download_uncompress_tar_gz(url, path='.', chunk_size=None):

    tmp_file = 'tmp.tar.gz'
    if chunk_size is None:
        chunk_size = 4 * 1024 * 1024

    # stream the data
    r = requests.get(url, stream=True)
    with open(tmp_file, 'wb') as f:
        content_length = int(r.headers['content-length'])
        count = 0
        for chunk in r.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            count += 1
            print("%d bytes out of %d downloaded" % 
                (count*chunk_size, content_length))
    r.close()

    # uncompress
    tar_file = 'tmp.tar'
    with open(tar_file, "wb") as f_u:
        with gzip.open(tmp_file, "rb") as f_c:
            f_u.write(f_c.read())

    # finally untar the file to final destination
    tf = tarfile.open(tar_file)

    if not os.path.exists(path):
        os.makedirs(path)
    tf.extractall(path)

    # remove the temporary file
    os.unlink(tmp_file)
    os.unlink(tar_file)



def modify_input_wav(wav,noise,room_dim,max_order,snr_vals):

    '''
    for mono
    '''

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
        max_order = max_order)

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
    if(len(noise_reverb[0]) < len(audio_reverb[0])):
        raise ValueError('the length of the noise signal is inferior to the one of the audio signal !!')

    #normalize the noise
    noise_reverb = noise_reverb[:,:len(audio_reverb[0])][0]
    noise_normalized = noise_reverb/np.linalg.norm(noise_reverb)

    noisy_signal = np.zeros((len(snr_vals),audio_reverb.shape[0], audio_reverb.shape[1]))

    for i,snr in enumerate(snr_vals):
        noise_std = np.linalg.norm(audio_reverb[0])/(10**(snr/20.))
        final_noise = noise_normalized*noise_std
        noisy_signal[i,:,:] = audio_reverb[0] + final_noise
    return noisy_signal