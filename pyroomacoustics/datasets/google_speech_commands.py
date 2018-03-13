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

            h = 1

            # TODO build corpus using `Meta` object and `add_sample` function.



