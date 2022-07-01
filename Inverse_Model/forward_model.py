
from tensorflow import keras

class forward_model:
    def __init__(self):
        self.model_path = "Models/model_cp.h5"
        self.pretrained_model = keras.models.load_model(self.model_path)
        self.model = self.pretrained_model

    def infer(self, data):
        output = self.model(data)
        output = output.numpy()
        return output

    def reset_model(self):
        self.model = self.pretrained_model
        if self.model == self.pretrained_model:
            return 1
        else:
            return 0
