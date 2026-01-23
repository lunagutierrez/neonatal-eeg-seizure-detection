import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

def evaluate(model, X, y):
    preds = np.argmax(model.predict(X), axis=1)

    return {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds),
        "recall": recall_score(y, preds),
    }