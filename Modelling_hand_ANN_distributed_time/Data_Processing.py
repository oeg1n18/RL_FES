import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


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

        input = np.reshape(input, (2520, 7, 251))
        output = np.reshape(output, (2520, 3, 251))
        return input, output

    def preprocess(self, input, output):
        N = input.shape[0]
        np.random.shuffle(input)
        np.random.shuffle(output)
        train_index = np.arange(np.ceil(0.8 * N)).astype('int')
        valid_index = np.arange(np.ceil(0.8 * N), np.ceil(0.92 * N)).astype('int')
        test_index = np.arange(np.ceil(0.92 * N), N).astype('int')
        self.X_train = np.swapaxes(input[train_index, :, :], 1, 2)
        self.Y_train = np.swapaxes(output[train_index, :, :], 1, 2)
        self.X_valid = np.swapaxes(input[valid_index, :, :], 1, 2)
        self.Y_valid = np.swapaxes(output[valid_index, :, :], 1, 2)
        self.X_test = np.swapaxes(input[test_index, :, :], 1, 2)
        self.Y_test = np.swapaxes(output[test_index, :, :], 1, 2)
        return self.X_train, self.Y_train, self.X_valid, self.Y_valid, self.X_test, self.Y_test

    def preprocess_forecast(self, X, Y, window_size):
        X_shape, Y_shape = X.shape, Y.shape
        Y_A = Y[:, :, 0]
        Y_B = Y[:, :, 1]
        Y_C = Y[:, :, 2]
        Y_A_timedist = np.empty((Y_A.shape[0], Y_A.shape[1] - window_size-1, window_size))
        Y_B_timedist = np.empty((Y_A.shape[0], Y_A.shape[1] - window_size-1, window_size))
        Y_C_timedist = np.empty((Y_A.shape[0], Y_A.shape[1] - window_size-1, window_size))

        for data_point in np.arange(Y_A.shape[0]):
            for time in np.arange(Y_A.shape[1] - window_size - 1):
                Y_A_timedist[data_point, time, :] = Y_A[data_point, time:time + window_size]
                Y_B_timedist[data_point, time, :] = Y_B[data_point, time:time + window_size]
                Y_C_timedist[data_point, time, :] = Y_C[data_point, time:time + window_size]

        X = np.delete(X, np.arange(X_shape[1] - (window_size+1), X_shape[1]), 1)
        return X, [Y_A_timedist, Y_B_timedist, Y_C_timedist]

    def scale_data(self, X, Y):
        X_scaled = np.empty(X.shape)
        Y_scaled = [np.empty(Y[0].shape), np.empty(Y[1].shape), np.empty(Y[2].shape)]
        X_maxes = []
        X_mins = []
        Y_maxes = []
        Y_mins = []
        for n in np.arange(X.shape[2]):
            X_maxes.append(np.max(X[:, :, n]))
            X_mins.append(np.min(X[:, :, n]))
            if X_maxes[n] != 0:
                X_scaled[:, :, n] = ((X[:, :, n] - X_mins[n]) / (X_maxes[n] - X_mins[n])) * 2 - 1
            else:
                X_scaled[:, :, n] = X[:, :, n]
        for n in np.arange(3):
            Y_maxes.append(np.max(Y[n]))
            Y_mins.append(np.min(Y[n]))
            if Y_maxes[n] != 0:
                Y_scaled[n] = ((Y[n] - Y_mins[n]) / (Y_maxes[n] - Y_mins[n])) * 2 - 1
            else:
                Y_scaled[n] = Y[n]
        return X_scaled, Y_scaled, (X_maxes, X_mins), (Y_maxes, Y_mins)


    def fit_transform(self):
        self.X_train, self.Y_train, self.X_valid, self.Y_valid, self.X_test, self.Y_test = self.preprocess(self.raw_input, self.raw_output)
        self.X_train, self.Y_train = self.preprocess_forecast(self.X_train, self.Y_train, self.forecast_horizon)
        self.X_valid, self.Y_valid = self.preprocess_forecast(self.X_valid, self.Y_valid, self.forecast_horizon)
        self.X_test, self.Y_test = self.preprocess_forecast(self.X_test, self.Y_test, self.forecast_horizon)
        self.X_train, self.Y_train, self.X_train_metrics, self.Y_train_metrics = self.scale_data(self.X_train, self.Y_train)
        self.X_valid, self.Y_valid, _, _ = self.scale_data(self.X_valid, self.Y_valid)
        self.X_test, self.Y_test, _, _ = self.scale_data(self.X_test, self.Y_test)
        return [(self.X_train, self.Y_train),
                (self.X_valid, self.Y_valid), (self.X_test, self.Y_test)]


    def unscale_data(self, Y):
        Y_unscaled = [np.empty(Y[0].shape), np.empty(Y[1].shape), np.empty(Y[2].shape)]
        for n in np.arange(len(Y)):
            Y_unscaled[n] = ((Y[n] + 1)/2)*(self.Y_train_metrics[0][n] - self.Y_train_metrics[1][n]) + self.Y_train_metrics[1][n]
        return Y_unscaled




def create_models(pipeline):
    model_ang1 = keras.models.Sequential([
        keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 7]),
        keras.layers.LSTM(20, return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Dense(pipeline.forecast_horizon, activation=keras.activations.tanh))
    ])

    model_ang2 = keras.models.Sequential([
        keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 7]),
        keras.layers.LSTM(20, return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Dense(pipeline.forecast_horizon, activation=keras.activations.tanh))
    ])

    model_ang3 = keras.models.Sequential([
        keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 7]),
        keras.layers.LSTM(20, return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Dense(pipeline.forecast_horizon, activation=keras.activations.tanh))
    ])
    return [model_ang1, model_ang2, model_ang3]


def compile_models(models, metric):
    for model in models:
        model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
    return models


def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])


def train_models(models, X_train, Y_train, X_valid, Y_valid, epochs):
    history = []
    for i, model in enumerate(models):
        history.append(model.fit(X_train, Y_train[i], epochs=epochs, validation_data=(X_valid, Y_valid[i])))
    return history, models


def save_training(models, history, model_root='Models/', training_root='training_loss/'):
    for i, model in enumerate(models):
        train_path = model_root + "model_ang" + str(i) + ".h5"
        model.save(train_path)
    for i, hist in enumerate(history):
        val_loss_path = training_root + "loss_model" + str(i) + "_validation" + ".csv"
        train_loss_path = training_root + "loss_model" + str(i) + "_train" + ".csv"
        np.savetxt(val_loss_path, hist.history["val_loss"], delimiter=",")
        np.savetxt(train_loss_path, hist.history["loss"], delimiter=",")


def test_accuracy(all_data, models):
    X_test, Y_test = all_data[2][0], all_data[2][1]
    for i, model in enumerate(models):
        print("Model" + str(i) + " Final MSE:  ")
        eval_score = model.evaluate(X_test, Y_test[i])


# ========================================================================

# models = create_models()
# models = compile_models(models, last_time_step_mse)
# history, models = train_models(models, data[0][0], data[1][0], data[0][1], data[1][1], 20)
# save_training(models, history)
# test_accuracy(data, models)

# model.compile(loss="mse", optimizer="adam")
# history = model.fit(X_train, Y_train, epochs=20,
#    validation_data=(X_valid, Y_valid))
