import numpy as np
import pandas as pd


class pipeline:
    def __init__(self):
        self.forecast_horizon = 10
        self.raw_input, self.raw_output = self.get_data()
        self.X_train = 0
        self.X_valid = 0
        self.X_test = 0
        self.X_train_metrics = 0
        self.Y_train = 0
        self.Y_valid = 0
        self.Y_test = 0
        self.Y_train_metrics = 0

    def get_data(self):
        input_df = pd.read_csv('Datasets/combinations_set/inputs.csv', header=None)
        output_df = pd.read_csv('Datasets/combinations_set/output.csv', header=None)

        input = input_df.to_numpy()
        output = output_df.to_numpy()

        n_input_elements = 0
        for dim in input.shape:
            if n_input_elements == 0:
                n_input_elements = dim
            else:
                n_input_elements = n_input_elements * dim

        n_output_elements = 0
        for dim in output.shape:
            if n_output_elements == 0:
                n_output_elements = dim
            else:
                n_output_elements = n_output_elements * dim

        n_input_samples = n_input_elements / (7 * 251)
        n_output_samples = n_output_elements / (3 * 251)
        input = np.reshape(input, (int(n_input_samples), 7, 251))
        output = np.reshape(output, (int(n_output_samples), 3, 251))
        return input, output

    def preprocess(self, input, output):
        N = input.shape[0]
        train_index = np.arange(np.ceil(0.8 * N)).astype('int')
        valid_index = np.arange(np.ceil(0.8 * N), np.ceil(0.92 * N)).astype('int')
        test_index = np.arange(np.ceil(0.92 * N), N).astype('int')
        self.X_train_full = np.swapaxes(input[train_index, :, :], 1, 2)
        self.Y_train_full = np.swapaxes(output[train_index, :, :], 1, 2)
        self.X_valid_full = np.swapaxes(input[valid_index, :, :], 1, 2)
        self.Y_valid_full = np.swapaxes(output[valid_index, :, :], 1, 2)
        self.X_test_full = np.swapaxes(input[test_index, :, :], 1, 2)
        self.Y_test_full = np.swapaxes(output[test_index, :, :], 1, 2)
        self.X_train = self.X_train_full[:, :-self.forecast_horizon, :]
        self.X_valid = self.X_valid_full[:, :-self.forecast_horizon, :]
        self.X_test = self.X_test_full[:, :-self.forecast_horizon, :]
        self.Y_train = self.Y_train_full[:, -self.forecast_horizon:, :]
        self.Y_valid = self.Y_valid_full[:, -self.forecast_horizon:, :]
        self.Y_test = self.Y_test_full[:, -self.forecast_horizon:, :]
        return self.X_train, self.Y_train, self.X_valid, self.Y_valid, self.X_test, self.Y_test

    def scale_data(self, X, Y):
        Y = np.nan_to_num(Y, nan=0.0)
        X = np.nan_to_num(X, nan=0.0)
        X_scaled = np.empty(X.shape)
        Y_scaled = np.empty(Y.shape)
        X_maxes = []
        X_mins = []
        Y_maxes = []
        Y_mins = []
        for n in np.arange(X.shape[1]):
            X_maxes.append(np.max(X[:, n, :]))
            X_mins.append(np.min(X[:, n, :]))
            if X_maxes[n] != 0:
                X_scaled[:, n, :] = ((X[:, n, :] - X_mins[n]) / (X_maxes[n] - X_mins[n]))
            else:
                X_scaled[:, n, :] = X[:, n, :]
        for n in np.arange(Y.shape[1]):
            Y_maxes.append(np.max(Y[:, n, :]))
            Y_mins.append(np.min(Y[:, n, :]))
            if Y_maxes[n] != 0:
                Y_scaled[:, n, :] = ((Y[:, n, :] - Y_mins[n]) / (Y_maxes[n] - Y_mins[n]))
            else:
                Y_scaled[:, n, :] = Y[:, n, :]
        return X_scaled, Y_scaled, (X_maxes, X_mins), (Y_maxes, Y_mins)

    def fit_transform(self):
        self.X_train_full, self.Y_train_full, self.X_metrics, self.Y_metrics = self.scale_data(self.raw_input,
                                                                                               self.raw_output)
        self.X_train, self.Y_train, self.X_valid, self.Y_valid, self.X_test, self.Y_test = self.preprocess(
            self.X_train_full, self.Y_train_full)
        return [(self.X_train, self.Y_train),
                (self.X_valid, self.Y_valid), (self.X_test, self.Y_test)]

    def unscale_data(self, Y):
        Y_unscaled = np.empty(Y.shape)
        for n in np.arange(Y.shape[2]):
            Y_unscaled[n] = ((Y[:, :, n] + 1) / 2) * (self.Y_train_metrics[0][n] - self.Y_train_metrics[1][n]) + \
                            self.Y_train_metrics[1][n]
        return Y_unscaled
