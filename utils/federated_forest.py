import numpy as np

class FederatedForest:

    def __init__(self):

        self.models = []

    def add_model(self, model):

        self.models.append(model)

    def predict(self, X):

        predictions = [model.predict(X) for model in self.models]
        predictions = np.array(predictions).T
        classes = np.unique(predictions)
        counts = [np.bincount(preds, minlength=max(classes)+1) for preds in predictions]
        y = classes[np.argmax(counts, axis=1)]
        # y = np.asarray(y)
        return y


if __name__ == "__main__":

    predictions = [np.random.randint(low=0, high=9, size=100) for _ in range(18)]
    predictions = np.array(predictions).T

    classes = np.unique(predictions)
    counts = [np.bincount(preds, minlength=max(classes)+1) for preds in predictions]
    y = classes[np.argmax(counts, axis=1)]

    print(y)
