import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras


def load_model(root="Models/", model_name="model.h5"):
    model_path = root + model_name
    trained_model = keras.models.load_model(model_path)
    return trained_model


def load_training_loss(root="training_loss/", training_losses=("loss_model.csv", "val_loss_model.csv")):
    train_losses = []
    for i, loss in enumerate(training_losses):
        path = root + loss
        train_losses.append(pd.read_csv(path, header=None).to_numpy())
    return train_losses


def plot_training_loss(train_losses):
    epochs = np.arange(0, train_losses[0].shape[0])
    fig = plt.figure()

    plt.plot(epochs, train_losses[0], label="training_loss_ang1")
    plt.plot(epochs, train_losses[1], label="validation_loss_ang1")
    plt.title("Angle1")
    plt.legend()

    plt.show()


def plot_prediction(model, pipeline):
    sample = np.random.randint(80)
    X_test, Y_test = pipeline.X_test, pipeline.Y_test

    preds = model(X_test)

    time = np.arange(249) * 0.02
    fig = plt.figure()

    fig.add_subplot(131)
    plt.plot(time, Y_test[sample, 1:, 0], label="True angle 1")
    plt.plot(time, preds[sample, :-1, 0], label="predicted angle 1")
    plt.title("Angle1")
    plt.ylim(-0.8, 1)
    plt.legend()

    fig.add_subplot(132)
    plt.plot(time, Y_test[sample, 1:, 1], label="True angle 2")
    plt.plot(time, preds[sample, :-1, 1], label="predicted angle 2")
    plt.title("Angle2")
    plt.ylim(-0.8, 1)
    plt.legend()

    fig.add_subplot(133)
    plt.plot(time, Y_test[sample, 1:, 2], label="True angle 3")
    plt.plot(time, preds[sample, :-1, 2], label="predicted angle 3")
    plt.title("Angle3")
    plt.ylim(-0.8, 1)

    plt.legend()

    plt.show()
