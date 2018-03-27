'''
The speech commands dataset consists of 65,000 WAVE audio files of people saying thirty different words.
This data was collected by Google and released under a CC BY license.
You can help omprove it by contributing with your own voice. The archive is over 1GB
'''

import os, glob
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.io import wavfile

from .utils import download_uncompress_tar_gz
from .base import Meta, AudioSample, Dataset


tensorflow_sounds = {
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
    'nine' : {'speech' : 1},
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

    def __init__(self, basedir=None, subset=None, download=False, build=True, 
        seed=13, **kwargs):

        # initialize
        Dataset.__init__(self)
        self.size_by_samples = {}

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
            self.build_corpus(subset, seed, **kwargs)


    def build_corpus(self, subset=None, seed=13, **kwargs):
        '''
        Build the corpus with some filters (speech or not speech)
        '''

        # # TODO
        # if subset is None:

        self.subdirs = glob.glob(os.path.join(self.basedir,'*','.'))
        self.classes = [s.split(os.sep)[-2] for s in self.subdirs]

        for idx, word in enumerate(self.classes):

            if word == '_background_noise_':
                speech = 0
            else:
                speech = 1

            word_path = self.subdirs[idx]

            self.size_by_samples[word] = 0
            for filename in glob.glob(os.path.join(word_path, '*.wav')):

                file_loc = os.path.join(self.basedir, word, os.path.basename(filename))

                # could also add score of original model for each word?
                if speech:
                    meta = Meta(word=word, speech=speech, file_loc=file_loc)
                else:
                    noise_type = os.path.basename(filename).split(".")[0]
                    meta = Meta(noise_type=noise_type, speech=speech, file_loc=file_loc)

                # not sure about this command
                if meta.match(**kwargs):
                    self.add_sample(GoogleSample(filename, **meta.as_dict()))

                self.size_by_samples[word] += 1
        # else:

        # 	if not isinstance(subset,int):
        # 		raise ValueError("the subset value has to be the size of the subset you want per words.")

        # 	self.subdirs = glob.glob(os.path.join(self.basedir,'*','.'))
	       #  self.classes = [s.split(os.sep)[-2] for s in self.subdirs]

	       #  for idx, word in enumerate(self.classes):

	       #      if word == '_background_noise_':
	       #          speech = 0
	       #      else:
	       #          speech = 1

	       #      word_path = self.subdirs[idx]

	       #      self.size_by_samples[word] = 0
	       #      for filename in glob.glob(os.path.join(word_path, '*.wav')):

	       #          file_loc = os.path.join(self.basedir, word, os.path.basename(filename))

	       #          # could also add score of original model for each word?
	       #          if speech:
	       #              meta = Meta(word=word, speech=speech, file_loc=file_loc)
	       #          else:
	       #              noise_type = os.path.basename(filename).split(".")[0]
	       #              meta = Meta(noise_type=noise_type, speech=speech, file_loc=file_loc)

        #             # not sure about this command
        #             if meta.match(**kwargs):
        #                 if (word == '_background_noise_'):
        #                 	self.add_sample(GoogleSample(filename, **meta.as_dict()))
        #                 else:
        #                 	if(self.size_by_samples[word] < subset):
        #                 		self.add_sample(GoogleSample(filename, **meta.as_dict()))

        #             self.size_by_samples[word] += 1



            # make sure subset is an integer and take `subset` values from each class
            # and all the background noise
            # something like this to get all the files for a particular class
            #files = glob.glob('google_speech_commands/seven/*.wav')
            #n_files = len(files)
            # same as in subset below
            # then loop through subset values




    def subset(self, n=10, trainTestSplit = False, seed=13):
        '''
        Build new corpus which are subset of a given size of the original one.
        The elements composing these new_corpus are taken randomly from the original corpus.

        We can use these functions to create new training and testing set form the full dataset.

        RETURN:
        -One new sample if trainTestSplit is false
        -Two new samples if trainTestSplit is true
        '''


        # DRAWING RANDOM VALUES --> TODO
        n_samples = 20
        subset_len = 5
        idx = np.arange(n_samples)
        np.random.seed(seed) # BEFORE SHUFFLING SET SEED, BEFORE EACH TIME YOU DO SHUFFLE!
        np.random.shuffle(idx)
        subset_idx = idx[:subset_len]
        ###

        

        indices = {}

        if(trainTestSplit):
            train_corpus = GoogleSpeechCommands()
            test_corpus = GoogleSpeechCommands()
            train_split = []
        else:
            new_corpus = GoogleSpeechCommands()

        
        for word in enumerate(self.classes):
            indices[word] = np.random.randint(0,self.size_by_samples[word]-1,n)

        for word in enumerate(self.classes):
            for index in indices[word]:
                if(trainTestSplit):
                    train_split.append(__getitem__(self,index))
                else:
                    new_corpus.add_sample(__getitem__(self,index))

        if(trainTestSplit):
            train,test = train_test_split(train_split,0.2,0.8)
            for sample in train:
                train_corpus.add_sample(sample)
            for sample in test:
                test_corpus.add_sample(sample)
            return (train_corpus,test_corpus)
        else:
            return new_corpus







        

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

        meta_dict = self.meta.as_dict()
        # if meta_dict['speech']:
        if self.meta.speech:
            template = 'speech: ''{speech}''; word: ''{word}''; file_loc: ''{file_loc}'''
        else:
            template = 'speech: ''{speech}''; noise type: ''{noise_type}''; file_loc: ''{file_loc}'''
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
        if self.meta.speech:
            plt.title(self.meta.file_loc)
        else:
            plt.title(self.meta.file_loc)

