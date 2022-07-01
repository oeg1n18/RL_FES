from forward_model import forward_model
from inverse_model import inverse_model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

EPOCHS = 300
BATCH_SIZE = 32
TRAIN_MODEL = True
MODEL_NAME = "inverse_model.h5"

forward_model = forward_model()
inverse_model = inverse_model()


def generate_ramp(N_input):
    input = np.zeros((N_input, 250, 7))
    ramp_metrics = np.zeros((N_input, 17))
    for batch in np.arange(N_input):
        channel1 = np.random.randint(8)
        channel2 = np.random.randint(8)
        amp1 = np.random.rand() * 0.15
        amp2 = np.random.rand() * 0.15
        width = int((0.5 + np.random.rand()) / 0.02)
        ramp = np.linspace(0, 1, width)

        ohe_channel1 = np.zeros(7).astype(np.float32)
        ohe_channel2 = np.zeros(7).astype(np.float32)
        if channel1 < 7 and channel2 == 7:
            input[batch, 25:25 + ramp.size, channel1] = ramp * amp1
            input[batch, ramp.size + 25:, channel1] = amp1
            ohe_channel1[channel1] = 1.0
        elif channel2 < 7 and channel1 == 7:
            input[batch, 25:25 + ramp.size, channel2] = ramp * amp2
            input[batch, ramp.size + 25:, channel2] = amp2
            ohe_channel2[channel2] = 1.0
        elif channel1 < 7 and channel2 < 7:
            input[batch, 25:25 + ramp.size, channel1] = ramp * amp1
            input[batch, 25:25 + ramp.size, channel2] = ramp * amp2
            input[batch, ramp.size + 25:, channel1] = amp1
            input[batch, ramp.size + 25:, channel2] = amp2
            ohe_channel1[channel1] = 1.0
            ohe_channel2[channel2] = 1.0
        ramp_metrics[batch, :] = np.hstack(
            (np.array([(50 + ramp.size * 0.02), amp1, amp2]), ohe_channel1, ohe_channel2))
    return input, ramp_metrics


def generate_ramp_from_inputs(ramp_metrics):
    inputs = np.zeros((ramp_metrics.shape[0], 250, 7))
    for input_i in np.arange(ramp_metrics.shape[0]):
        ramp_metric = ramp_metrics[int(input_i), :]
        stop_time = ramp_metric[0]
        amp1 = ramp_metrics[1]
        amp2 = ramp_metrics[2]
        channel1 = np.argmax(ramp_metrics[3:10])
        channel2 = np.argmax(ramp_metrics[11:])
        width = int(stop_time/0.02 - 25)
        ramp = np.linspace(0, 1, width)
        if channel1 < 7 and channel2 == 7:
            inputs[int(input_i), 25:25 + ramp.size, channel1] = ramp * amp1
            inputs[int(input_i), ramp.size + 25:, channel1] = amp1
        elif channel2 < 7 and channel1 == 7:
            inputs[int(input_i), 25:25 + ramp.size, channel2] = ramp * amp2
            inputs[int(input_i), ramp.size + 25:, channel2] = amp2
        elif channel1 < 7 and channel2 < 7:
            inputs[int(input_i), 25:25 + ramp.size, channel1] = ramp * amp1
            inputs[int(input_i), 25:25 + ramp.size, channel2] = ramp * amp2
            inputs[int(input_i), ramp.size + 25:, channel1] = amp1
            inputs[int(input_i), ramp.size + 25:, channel2] = amp2
    return inputs


if TRAIN_MODEL:
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath="Inverse_Models/my_best_model.epoch{epoch:02d}-loss{loss:.2f}.hdf5")
    callbacks = [checkpoint]
    for epoch in np.arange(EPOCHS):
        print(epoch)
        inputs, ramp_metrics = generate_ramp(BATCH_SIZE)
        outputs = forward_model.infer(inputs)
        print(outputs.shape, ramp_metrics.shape)
        inverse_model.fit(outputs, ramp_metrics, callbacks)
    keras.models.save_model(inverse_model.model, "Inverse_Models/inverse_model.h5")

if not TRAIN_MODEL:
    inverse_model.model = keras.models.load_model("Inverse_Models/" + MODEL_NAME)

X_test = generate_ramp(50)
Y_test = forward_model.infer(X_test)
Y_preds = inverse_model.infer(Y_test)
inverse_model.model.evaluate(Y_test, X_test)

fig = plt.figure()
fig.add_subplot(121)
plt.plot(X_test[1, :, :])
plt.title("Actual Inputs")

fig.add_subplot(122)
plt.plot(Y_preds[1, :, :])
plt.title("Predictied Inputs")

plt.show()
