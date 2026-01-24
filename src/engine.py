import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from cnn import build_cnn
from config import KFOLDS, SEED, EPOCHS, BATCH_SIZE
import time
from logger import get_logger
from evaluation import MetricsPerEpoch
import math
from tensorflow.keras.callbacks import LearningRateScheduler
import pandas as pd
from pathlib import Path

logger = get_logger(__name__)

def drop_based_step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.75
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate

def run_kfold_experiment(x: np.ndarray, y: np.ndarray, expert: str, window: int, chunk: int, out_dir: Path):
    """K-Fold training with per-epoch averaged confusion metrics."""
    
    kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=SEED)
    all_epoch_data_folds = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(x)):
        fold_start = time.time()
        logger.info(f"[Fold {fold+1}/{KFOLDS}] Building model")

        model, _ = build_cnn(input_shape=x.shape[1:])
        lr_scheduler = LearningRateScheduler(drop_based_step_decay, verbose=0)

        metrics_callback = MetricsPerEpoch(
            X_val=x[val_idx],
            y_val=y[val_idx],
            X_train=x[train_idx],
            y_train=y[train_idx]
        )

        logger.info(f"[Fold {fold+1}/{KFOLDS}] Training")
        model.fit(
            x[train_idx], y[train_idx],
            validation_data=(x[val_idx], y[val_idx]),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[lr_scheduler, metrics_callback],
            verbose=0
        )

        # Add this fold's metrics
        all_epoch_data_folds.append(pd.DataFrame(metrics_callback.epoch_data))
        tf.keras.backend.clear_session()
        fold_time = time.time() - fold_start
        logger.info(f"[Fold {fold+1}/{KFOLDS}] Done | Time={fold_time:.1f}s")

    # --- Average per epoch across folds ---
    combined = pd.concat(all_epoch_data_folds)
    avg_epoch_data = combined.groupby(["epoch", "dataset"]).mean().reset_index()

    # Save averaged per-epoch metrics
    epoch_metrics_file = out_dir / f"epoch_metrics_avg_expert_{expert}_W{window}_C{chunk}.csv"
    Path(epoch_metrics_file.parent).mkdir(parents=True, exist_ok=True)
    avg_epoch_data.to_csv(epoch_metrics_file, index=False)

    # Return average accuracy over all folds/last epoch for summary table
    last_epoch_val = avg_epoch_data[avg_epoch_data["dataset"] == "val"].iloc[-1]
    avg_metrics = {
        "accuracy": last_epoch_val["accuracy"],
        "TP": last_epoch_val["TP"],
        "TN": last_epoch_val["TN"],
        "FP": last_epoch_val["FP"],
        "FN": last_epoch_val["FN"]
    }

    return avg_metrics
