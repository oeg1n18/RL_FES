from tensorflow import keras
import numpy as np
import os


def create_model():
    model = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=[None, 7]),
    keras.layers.LSTM(35, return_sequences=True),
    keras.layers.LSTM(35, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(3))])
    return model


def train_models(model, pipeline, epochs):
    checkpoint_path = os.getcwd() + "/Models/model_cp.h5"
    save_cp = keras.callbacks.ModelCheckpoint(checkpoint_path)
    print(pipeline.X_train.shape, pipeline.Y_train.shape)
    print(pipeline.X_valid.shape, pipeline.Y_valid.shape)
    history = model.fit(pipeline.X_train, pipeline.Y_train, epochs=epochs,
                        validation_data=(pipeline.X_valid, pipeline.Y_valid), callbacks=[save_cp])
    return history, model


def save_training(model, history, model_root='Models/', training_root='training_loss/'):
        train_path = model_root + "model.h5"
        model.save(train_path)
        val_loss_path = training_root + "val_loss_model.csv"
        train_loss_path = training_root + "loss_model.csv"
        np.savetxt(val_loss_path, history.history["val_loss"], delimiter=",")
        np.savetxt(train_loss_path, history.history["loss"], delimiter=",")
