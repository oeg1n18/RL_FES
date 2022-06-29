import numpy as np
from tensorflow import keras
import pandas as pd
from data_pipeline import *
from Training import *
from evaluate_models import *


def train_all_models(pipeline, epochs, new_model=False):
    pipeline.fit_transform()
    if new_model:
        model = create_model()
    else:
        model = load_model()
    print(model.summary())
    model.compile(loss="mse", optimizer="adam")
    history, models = train_models(model, pipeline, epochs)
    save_training(model, history)
    return pipeline


def evaluate_models(type, pipeline):
    if type == "Loss":
        train_losses = load_training_loss()
        plot_training_loss(train_losses)
    elif type == "Graph":
        models = load_model()
        plot_prediction(models, pipeline)
    else:
        print("Please choose Loss or Graph for type")


def get_accuracy(pipeline):
    model = load_model()
    X_test, Y_test = pipeline.X_test, pipeline.Y_test
    preds = model(X_test)
    error = np.abs(Y_test - preds)
    return 1 - np.mean(error)




