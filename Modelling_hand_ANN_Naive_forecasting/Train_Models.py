import numpy as np
from tensorflow import keras
import pandas as pd
from data_pipeline import *
from Training import *
from evaluate_models import *


def train_all_models(pipeline, epochs, horizon):
    pipeline.forecast_horizon = horizon
    pipeline.fit_transform()
    models = create_models(pipeline)
    models = compile_models(models)
    history, models = train_models(models, pipeline, epochs)
    save_training(models, history)
    return pipeline


def evaluate_models(type, pipeline):
    if type == "Loss":
        train_losses = load_training_loss()
        plot_training_loss(train_losses)
    elif type == "Graph":
        models = load_models()
        plot_prediction(models, pipeline)
    else:
        print("Please choose Loss or Graph for type")


def get_accuracy(pipeline):
    models = load_models()
    X_test, Y_test = pipeline.X_test, pipeline.Y_test
    Y_predictions = []
    for feature in np.arange(Y_test.shape[2]):
        Y_predictions.append(np.mean(models[feature](X_test), axis=1))

    Y_true = []
    for feature in np.arange(Y_test.shape[2]):
        Y_true.append(np.mean(Y_test[:, :, feature], axis=1))

    for feature in np.arange(Y_test.shape[2]):
        accuracy = 1.0 - np.mean(np.abs(Y_true[feature] - Y_predictions[feature]))
        print("model_ang" + str(feature) + " accuracy: " + str(accuracy))



pipeline = pipeline()
pipeline.forecast_horizon = 10
pipeline.fit_transform()
#pipeline = train_all_models(pipeline, 30, 10)
#get_accuracy(pipeline)
evaluate_models("Graph", pipeline)


