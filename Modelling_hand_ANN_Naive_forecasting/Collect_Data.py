


import numpy as np
import matlab.engine
import csv
import warnings
import os

warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')
eng = matlab.engine.start_matlab()

n_samples = 3000

with open('Datasets/combinations_set/inputs.csv', 'w', newline='') as input_file, open('Datasets/combinations_set/output.csv', 'w', newline='') as output_file:
    input_writer = csv.writer(input_file)
    output_writer = csv.writer(output_file)
    i = 0
    while i < n_samples:
        hand_input, hand_output = eng.hand_test(nargout=2)
        hand_input = np.array(hand_input)
        hand_output = np.array(hand_output)
        for chan in np.arange(7):
            input_writer.writerow(hand_input[chan, :])
        for chan in np.arange(3):
            output_writer.writerow(hand_output[chan, :])
        os.system('clear')
        i += 1
        print("completed test " + str(i) + " of " + str(n_samples))





