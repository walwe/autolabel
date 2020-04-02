
class Classifier:

    NAME = None

    def __init__(self):
        self._model = None

    def predict(self, images, decode, top):
        raise NotImplementedError

    def decode(self, preds, top=3):
        raise NotImplementedError
