'''
Google Speech Commands Dataset
=================

Pyroomacoustics includes a wrapper around the Google Speech Commands dataset [TODO add reference].
'''

import sys
sys.path.append("../")

import pyroomacoustics as pra
import os, argparse

if __name__ == '__main__':

    # create object
    dataset = pra.datasets.GoogleSpeechCommands(download=True)

