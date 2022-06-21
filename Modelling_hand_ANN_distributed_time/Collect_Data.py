


import numpy as np
import tensorflow as tf
import matlab.engine
import csv
import itertools
import warnings
import os

warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')
eng = matlab.engine.start_matlab()


with open('Datasets/combinations_set/inputs.csv', 'w', newline='') as input_file, open('Datasets/combinations_set/output.csv', 'w', newline='') as output_file:
    input_writer = csv.writer(input_file)
    output_writer = csv.writer(output_file)
    i = 0
    for channels in list(itertools.permutations(np.arange(7), 5)):
        hand_input, hand_output = eng.hand_test(int(channels[0]), int(channels[1]), int(channels[2]), int(channels[3]), int(channels[4]), nargout=2)
        hand_input = np.array(hand_input)
        hand_output = np.array(hand_output)
        for chan in np.arange(7):
            input_writer.writerow(hand_input[chan, :])
        for chan in np.arange(3):
            output_writer.writerow(hand_output[chan, :])
        os.system('clear')
        i = i+1
        print("completed test " + str(i) + " of 2520")





