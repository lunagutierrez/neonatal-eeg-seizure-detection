import tensorflow as tf
import numpy as np

class MetricsPerEpoch(tf.keras.callbacks.Callback):
    """
    Records per-epoch confusion metrics for train and validation.
    """
    def __init__(self, X_val, y_val, X_train=None, y_train=None):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.X_train = X_train
        self.y_train = y_train
        self.epoch_data = []

    @staticmethod
    def compute_confusion(y_true, y_pred):
        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        return TP, TN, FP, FN, accuracy

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # --- Training metrics ---
        if self.X_train is not None and self.y_train is not None:
            y_pred_train = (self.model.predict(self.X_train, verbose=0) > 0.5).astype(int).flatten()
            TP, TN, FP, FN, acc = self.compute_confusion(self.y_train, y_pred_train)
            loss_train = logs.get("loss")

            self.epoch_data.append({
                "dataset": "train",
                "epoch": epoch + 1,
                "loss": loss_train,
                "TP": TP,
                "TN": TN,
                "FP": FP,
                "FN": FN,
                "accuracy": acc
            })

        # --- Validation metrics ---
        y_pred_val = (self.model.predict(self.X_val, verbose=0) > 0.5).astype(int).flatten()
        TP, TN, FP, FN, acc = self.compute_confusion(self.y_val, y_pred_val)
        loss_val = self.model.evaluate(self.X_val, self.y_val, verbose=0)[0]

        self.epoch_data.append({
            "dataset": "val",
            "epoch": epoch + 1,
            "loss": loss_val,
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN,
            "accuracy": acc
        })
