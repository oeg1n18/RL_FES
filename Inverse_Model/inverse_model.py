from tensorflow import keras


class inverse_model:
    def __init__(self):
        self.model = keras.models.Sequential([
                    keras.layers.InputLayer(input_shape=[None, 3]),
                    keras.layers.LSTM(35, return_sequences=True),
                    keras.layers.LSTM(35),
                    keras.layers.Dense(17)])
        self.model.compile(loss="mse", optimizer="adam")
        self.losses = []

    def infer(self, data):
        output = self.model(data)
        output = output.numpy()
        return output

    def fit(self, x, y, callbacks):
        history = self.model.fit(x, y, epochs=1, callbacks=callbacks)
        loss = history.history["loss"]
        self.losses.append(loss[-1:])


