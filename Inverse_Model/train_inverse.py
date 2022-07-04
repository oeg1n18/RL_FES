from forward_model import forward_model
from inverse_model import inverse_model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

EPOCHS = 60
BATCH_SIZE = 500
TRAIN_MODEL = False
MODEL_NAME = "my_best_model.h5"
LOAD_DATA = True

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
        #print("Sent ", channel1, " ", channel2, " ", amp1, " ", amp2, " ", int(ramp.size))

        ohe_channel1 = np.zeros(7).astype(np.float32)
        ohe_channel2 = np.zeros(7).astype(np.float32)
        if channel1 < 7 and channel2 >= 7:
            input[batch, 25:25 + ramp.size, channel1] = ramp * amp1
            input[batch, ramp.size + 25:, channel1] = amp1
            ohe_channel1[channel1] = 1.0
        elif channel2 < 7 and channel1 >= 7:
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
            (np.array([ramp.size*0.01, amp1, amp2]), ohe_channel1, ohe_channel2))
    return input, ramp_metrics


def generate_ramp_from_inputs(ramp_metrics):
    inputs = np.zeros((ramp_metrics.shape[0], 250, 7))
    for input_i in np.arange(ramp_metrics.shape[0]):
        width = int(ramp_metrics[int(input_i), 0]*100)
        amp1 = ramp_metrics[int(input_i), 1]
        amp2 = ramp_metrics[int(input_i), 2]
        channel1 = np.argmax(ramp_metrics[int(input_i), 3:10])
        if np.max(ramp_metrics[int(input_i), 3:10]) == 0:
            channel1 = 7
        channel2 = np.argmax(ramp_metrics[int(input_i), 10:])
        if np.max(ramp_metrics[int(input_i), 10:]) == 0:
            channel2 = 7
        #print("Received: ", channel1, " ", channel2, " ", amp1, " ", amp2, " ", width)
        if width < 0.5/0.02:
            width = 25
        if width > 1.5/0.02:
            width = 75
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
        inputs, ramp_metrics = generate_ramp(BATCH_SIZE)
        outputs = forward_model.infer(inputs)
        inverse_model.fit(outputs, ramp_metrics, callbacks)
    keras.models.save_model(inverse_model.model, "Inverse_Models/inverse_model.h5")

if not TRAIN_MODEL:
    inverse_model.model = keras.models.load_model("Inverse_Models/inverse_model.h5")

X_test, test_metrics = generate_ramp(3)
#outputs = forward_model.infer(X_test)
#Y_preds_metrics = inverse_model.infer(outputs)

pred_inputs = generate_ramp_from_inputs(test_metrics)


#inverse_model.model.evaluate(Y_test, Y_preds_metrics)
sample = np.random.randint(3)

print("Sample:", sample)

#print("===========================")



fig = plt.figure()
fig.add_subplot(121)
plt.plot(X_test[sample, :, 0], label="Channel0")
plt.plot(X_test[sample, :, 1], label="Channel1")
plt.plot(X_test[sample, :, 2], label="Channel2")
plt.plot(X_test[sample, :, 3], label="Channel3")
plt.plot(X_test[sample, :, 4], label="Channel4")
plt.plot(X_test[sample, :, 5], label="Channel5")
plt.plot(X_test[sample, :, 6], label="Channel6")
plt.legend()
plt.title("Actual Inputs")

fig.add_subplot(122)
plt.plot(pred_inputs[sample, :, 0], label="Channel0")
plt.plot(pred_inputs[sample, :, 1], label="Channel1")
plt.plot(pred_inputs[sample, :, 2], label="Channel2")
plt.plot(pred_inputs[sample, :, 3], label="Channel3")
plt.plot(pred_inputs[sample, :, 4], label="Channel4")
plt.plot(pred_inputs[sample, :, 5], label="Channel5")
plt.plot(pred_inputs[sample, :, 6], label="Channel6")
plt.legend()
plt.title("Predictied Inputs")

plt.show()
