import numpy as np
import pandas as pd
from tensorflow import keras
from Data_Processing import *
from training import *

# build the pipeline
pipeline = pipeline()
pipeline.forecast_horizon = 10

# Process the data through the pipeline
pipeline.fit_transform()
#print(pipeline.X_train.shape, pipeline.Y_train[:, :, 1].shape)

# Create the models
models = create_models(pipeline)

# Compile the models
models = compile_models(models, last_time_step_mse)

# Train the models



history, trained_models = train_models(models, pipeline.X_train, pipeline.Y_train,
                                       pipeline.X_valid, pipeline.Y_valid, 10)

# Save the Models and training data
save_training(models, history)


#models = load_models()
#training_loss = load_training_loss()
#plot_training_loss(training_loss)
#plot_prediction(models, pipeline)




