import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import pandas as pd
from Data_Processing import *

def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])


def get_config(self):
    base_config = super().get_config()
    return {**base_config, "num_classes": self.num_classes}


def load_models(root="Models/", pretrained_models=("model_ang0.h5", "model_ang1.h5", "model_ang2.h5")):
    trained_models = []
    for i, model_name in enumerate(pretrained_models):
        model_path = root + model_name
        trained_models.append(
            keras.models.load_model(model_path, custom_objects={'last_time_step_mse': last_time_step_mse}))
    return trained_models


def load_training_loss(root="training_loss/", training_losses=("loss_model0_train.csv", "loss_model0_validation.csv",
                                                               "loss_model1_train.csv", "loss_model1_validation.csv",
                                                               "loss_model2_train.csv", "loss_model2_validation.csv")):
    train_losses = []
    for i, loss_name in enumerate(training_losses):
        path = root + loss_name
        train_losses.append(pd.read_csv(path, header=None).to_numpy())
    return train_losses


def plot_training_loss(train_losses):
    epochs = np.arange(1, 1 + train_losses[0].shape[0])
    fig = plt.figure()

    fig.add_subplot(131)
    plt.plot(epochs, np.transpose(train_losses[0])[0], label="training_loss_ang1")
    plt.plot(epochs, np.transpose(train_losses[1])[0], label="validation_loss_ang1")
    plt.title("Angle1")
    plt.legend()

    fig.add_subplot(132)
    plt.plot(epochs, np.transpose(train_losses[2])[0], label="training_loss_ang2")
    plt.plot(epochs, np.transpose(train_losses[3])[0], label="validation_loss_ang2")
    plt.title("Angle2")
    plt.legend()

    fig.add_subplot(133)
    plt.plot(epochs, np.transpose(train_losses[4])[0], label="training_loss_ang3")
    plt.plot(epochs, np.transpose(train_losses[5])[0], label="validation_loss_ang3")
    plt.title("Angle3")
    plt.legend()

    plt.show()




def plot_prediction(models, pipeline):
    sample = 1
    #input, output = pipeline.raw_input, pipeline.raw_output
    #_, _, _, _, X_test, Y_test = pipeline.preprocess(pipeline.raw_input, pipeline.raw_output)
    #X_test, Y_test = pipeline.preprocess_forecast(X_test, Y_test, pipeline.forecast_horizon)
    X_test, Y_test = pipeline.X_test, pipeline.Y_test

    Y_pred = []
    for i in np.arange(len(Y_test)):
        Y_pred.append(models[i](X_test))

    Y_pred_angle = []
    Y_true = []
    for i in np.arange(len(Y_pred)):
        pred = np.zeros((251,1))
        true = np.zeros((251, 1))
        window = 0
        while window < X_test.shape[1] - pipeline.forecast_horizon:
            window_pred = Y_pred[i][sample, int(window), :]
            pred[int(window):int(window) + pipeline.forecast_horizon] = np.reshape(window_pred, (pipeline.forecast_horizon, 1))
            window_true = Y_test[sample][sample, int(window), :]
            true[int(window):int(window) + pipeline.forecast_horizon] = np.reshape(window_true, (pipeline.forecast_horizon, 1))
            window += pipeline.forecast_horizon
        Y_true.append(true)
        Y_pred_angle.append(pred)

    time = np.arange(251) * 0.02
    print("hello")
    print(Y_true[0][-1])
    print(time.shape)
    print(Y_pred_angle[0].reshape((251,))[-1])


    fig = plt.figure()

    fig.add_subplot(131)
    plt.plot(time, Y_true[0], label="True_ang1")
    plt.plot(time, Y_pred_angle[0], label="Pred_ang1")
    plt.title("Angle1")
    plt.legend()

    fig.add_subplot(132)
    plt.plot(time, Y_true[1], label="True_ang2")
    plt.plot(time, Y_pred_angle[1], label="Pred_ang2")
    plt.title("Angle2")
    plt.legend()

    fig.add_subplot(133)
    plt.plot(time, Y_true[2], label="True_ang3")
    plt.plot(time, Y_pred_angle[2], label="Pred_ang3")
    plt.title("Angle3")
    plt.legend()

    plt.show()


