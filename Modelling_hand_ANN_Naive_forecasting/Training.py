from tensorflow import keras
from data_pipeline import pipeline
import numpy as np


def create_models(pipeline):
    models = []
    for n in np.arange(pipeline.Y_train.shape[2]):
        models.append(keras.models.Sequential([
            keras.layers.LSTM(30, return_sequences=True, input_shape=[None, 7]),
            keras.layers.LSTM(30),
            keras.layers.Dense(pipeline.forecast_horizon)
        ]))
    print(models[0].summary())
    return models


def compile_models(models):
    for model in models:
        model.compile(loss="mse", optimizer="adam")
    return models


def train_models(models, pipeline, epochs):
    histories = []
    for i, model in enumerate(models):
        history = model.fit(pipeline.X_train, pipeline.Y_train[:, :, i], epochs=epochs,
                            validation_data=(pipeline.X_valid, pipeline.Y_valid[:, :, i]))
        histories.append(history)
    return histories, models


def save_training(models, history, model_root='Models/', training_root='training_loss/'):
    for i, model in enumerate(models):
        train_path = model_root + "model_ang" + str(i) + ".h5"
        model.save(train_path)
    for i, hist in enumerate(history):
        val_loss_path = training_root + "loss_model" + str(i) + "_validation" + ".csv"
        train_loss_path = training_root + "loss_model" + str(i) + "_train" + ".csv"
        np.savetxt(val_loss_path, hist.history["val_loss"], delimiter=",")
        np.savetxt(train_loss_path, hist.history["loss"], delimiter=",")
