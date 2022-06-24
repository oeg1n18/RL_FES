import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras


def load_models(root="Models/", pretrained_models=("model_ang0.h5", "model_ang1.h5", "model_ang2.h5")):
    trained_models = []
    for i, model_name in enumerate(pretrained_models):
        model_path = root + model_name
        trained_models.append(keras.models.load_model(model_path))
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
    epochs = np.arange(0, train_losses[0].shape[0])
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
    sample = 60
    X_test, Y_test = pipeline.X_test, pipeline.Y_test

    Y_predictions = []
    for i in np.arange(Y_test.shape[2]):
        Y_predictions.append(models[i](X_test))

    Y_pred = []
    Y_true = []
    for feature in np.arange(3):
        Y_true.append(Y_test[sample, :, feature])
        Y_pred.append(Y_predictions[feature][sample, :])


    time = np.arange(251) * 0.02
    fig = plt.figure()

    print(pipeline.Y_test_full[sample, :-pipeline.forecast_horizon, 0].shape)
    print(Y_true[0].shape)
    print(Y_pred[0].shape)

    fig.add_subplot(131)
    plt.plot(time[:-pipeline.forecast_horizon], pipeline.Y_test_full[sample, :-pipeline.forecast_horizon, 0], label="True")
    #plt.plot(time[-pipeline.forecast_horizon:], Y_true[0],marker="o", markersize=10, markeredgecolor="green",label="True_ang1")
    plt.plot(time[-pipeline.forecast_horizon:], Y_pred[0],marker="o", markersize=10, markeredgecolor="red", label="Pred_ang1")
    plt.title("Angle1")
    plt.ylim(0, 1)
    plt.legend()

    fig.add_subplot(132)
    plt.plot(time[:-pipeline.forecast_horizon], pipeline.Y_test_full[sample, :-pipeline.forecast_horizon, 1], label="True")
    #plt.plot(time[-pipeline.forecast_horizon:], Y_true[1],marker="o", markersize=10, markeredgecolor="green",label="True_ang2")
    plt.plot(time[-pipeline.forecast_horizon:], Y_pred[1],marker="o", markersize=10, markeredgecolor="red", label="Pred_ang2")
    plt.title("Angle2")
    plt.ylim(0, 1)
    plt.legend()

    fig.add_subplot(133)
    plt.plot(time[:-pipeline.forecast_horizon], pipeline.Y_test_full[sample, :-pipeline.forecast_horizon, 2], label="True")
    plt.plot(time[-pipeline.forecast_horizon:], Y_true[2],marker="o", markersize=10, markeredgecolor="green",label="True_ang3")
    plt.plot(time[-pipeline.forecast_horizon:], Y_pred[2],marker="o", markersize=10, markeredgecolor="red", label="Pred_ang3")
    plt.title("Angle3")
    plt.ylim(0, 1)
    plt.legend()

    plt.show()