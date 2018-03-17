'''
TODO : info about dataset
'''

import os
import numpy as np
from scipy.io import wavfile

from .utils import download_uncompress_tar_gz
from .base import Meta, AudioSample, Dataset


tenserflow_sounds = {
    'zero' : { 'speech' : 1},
    'yes': {'speech' : 1},
    'wow': {'speech' : 1},
    'up' : {'speech' : 1},
    'two' : {'speech' : 1},
    'tree' : {'speech' : 1},
    'stop' : {'speech' : 1},
    'six' : {'speech' : 1},
    'sheila' : {'speech' : 1},
    'seven' : {'speech' : 1},
    'right' : {'speech' : 1},
    'one' : {'speech' : 1},
    'on' : {'speech' : 1},
    'off' : {'speech' : 1},
    'no' : {'speech' : 1},
    'nibe' : {'speech' : 1},
    'marvin' : {'speech' : 1},
    'left' : {'speech' : 1},
    'house' : {'speech' : 1},
    'happy' : {'speech' : 1},
    'go' : {'speech' : 1},
    'four' : {'speech' : 1},
    'five' : {'speech' : 1},
    'eight' : {'speech' : 1},
    'down' : {'speech' : 1},
    'dog' : {'speech' : 1},
    'cat' : {'speech' : 1},
    'bird' : {'speech' : 1},
    'bed' : {'speech' : 1},
    '_background_noise_' : {'speech' : 0},
    }


url = "http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz"

tenserflow_sounds_data = {}


class GoogleSpeechCommands(Dataset):
    '''

    Parameters
    ----------
    basedir: str, optional
        The directory where the Google Speech Command dataset is located/downloaded. By
        default, this is the current directory.
    download: bool, optional
        If the corpus does not exist, download it.
    build: bool, optional
        Can be 'female' or 'male'
    '''

    def __init__(self, basedir=None, download=False, build=True, **kwargs):

        # initialize
        Dataset.__init__(self)
        self.size_by_samples = {
            'zero' : 0,
            'yes': 0,
            'wow': 0,
            'up' : 0,
            'two' : 0,
            'tree' : 0,
            'stop' : 0,
            'six' : 0,
            'sheila' : 0,
            'seven' : 0,
            'right' : 0,
            'one' : 0,
            'on' : 0,
            'off' : 0,
            'no' : 0,
            'nibe' : 0,
            'marvin' : 0,
            'left' : 0,
            'house' : 0,
            'happy' : 0,
            'go' : 0,
            'four' : 0,
            'five' : 0,
            'eight' : 0,
            'down' : 0,
            'dog' : 0,
            'cat' : 0,
            'bird' : 0,
            'bed' : 0,
            '_background_noise_' : 0,
            } 

        # default base directory is the current one
        self.basedir = basedir
        if basedir is None:
            self.basedir = './google_speech_commands'

        # check the directory exists and download otherwise
        if not os.path.exists(self.basedir):
            if download:
                print('Downloading', url, 'into', self.basedir, '...')
                download_uncompress_tar_gz(url=url, path=self.basedir)
            else:
                raise ValueError('Dataset directory does not exist. Create or set download option.')
        else:
            print("Dataset exists! Using %s" % self.basedir)


        if build:
            self.build_corpus(**kwargs)


    def build_corpus(self, **kwargs):
        '''
        Build the corpus with some filters (speech or not speech)
        '''

        for word in tenserflow_sounds.keys():

            with open(os.path.join(self.basedir,'testing_list.txt'),'r') as f:
                if word is '_background_noise_':
                    speech = 0
                else:
                    speech = 1


                for line in f.readlines():
                    l = line.split('/')
                    sound = l[0]
                    file = l[1]
                    path = os.path.join(self.basedir,sound + '/' + file)

                    if sound not in tenserflow_sounds_data:
                        tenserflow_sounds_data[sound] ={
                            'speech' = speech
                            'paths' = np.array([path])
                        }
                    else:
                        np.append(tenserflow_sounds_data[sound]['paths'],[path])


        for word, info in tenserflow_sounds_data.items():
            for path in info['paths']:
                meta = Meta(speech = info['speech'], word = word) 
                if meta.match(**kwargs):
                    self.add_sample(GoogleSample(path, **meta.as_dict()))
                    self.size_by_samples[word] += 1

    def subset(self,size):
        select_list = []
        for word in tenserflow_sounds:
            r = filter(self,'word == word')
            for sample in r.samples[:size]:
                select_list.append(sample)
        r.build_corpus(self,**kwargs)


        

class GoogleSample(AudioSample):
    '''
    Create the sound object

    Parameters
    ----------
    path: str
      the path to the audio file
    **kwargs:
      metadata as a list of keyword arguments
    Attributes
    ----------
    data: array_like
      the actual audio signal
    fs: int
      sampling frequency
    '''

    def __init__(self,path,**kwargs):
        '''
        Create the the sound object
        path: string
          the path to a particular sample
        '''

        fs,data = wavfile.read(path)
        AudioSample.__init__(self, data, fs, **kwargs)

    def __str__(self):
        '''string representation'''

        template = '{word}: ''{speech}'''
        s = template.format(**self.meta.as_dict())
        return s


    def plot(self,**kwargs):
        '''Plot the spectogram'''
        try:
            import matplotlib.pyplot as plt 
        except ImportError:
            print('Warning: matplotlib is required for plotting')
            return
        AudioSample.plot(self,**kwargs)
        plt.title(self.meta.sound)