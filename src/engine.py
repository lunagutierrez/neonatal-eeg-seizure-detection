import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from  cnn import build_cnn
from  config import KFOLDS, SEED, EPOCHS, BATCH_SIZE
import time
from  logger import get_logger
logger = get_logger(__name__)

   
def run_kfold_experiment(x: np.ndarray, y: np.ndarray):
    """Executes k-fold cross-validation and returns average accuracy."""

    accuracies = []
    kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=SEED) # K-Fold splitter w/ shuffling before split

    for fold, (train_idx, val_idx) in enumerate(kf.split(x)):
        fold_start = time.time()
        logger.info(f"Fold {fold}/{KFOLDS} Building model")

        # Build fresh model for each fold to avoid weight leakage
        model, _ = build_cnn(input_shape=x.shape[1:]) 
        
        logger.info(f"[Fold {fold}/{KFOLDS}] Training")
        # Train the model
        model.fit(
            x[train_idx], y[train_idx],
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0
        )

        logger.info(f"[Fold {fold}/{KFOLDS}] Evaluating")
        # Evaluation using the 0.5 threshold for sigmoid output
        raw_preds = model.predict(x[val_idx], verbose=0) # Predictions on validation set
        preds = (raw_preds > 0.5).astype("int32") # Binarize predictions w/ threshold 0.5
        # Accuracy per fold
        acc = accuracy_score(y[val_idx], preds)
        accuracies.append(acc)
        
        # Memory cleanup = freed
        tf.keras.backend.clear_session()

        fold_time = time.time() - fold_start
        logger.info(f"[Fold {fold}/{KFOLDS}] Done | Acc={acc:.3f} | Time={fold_time:.1f}s")

    return np.mean(accuracies) # Avg accuracy across folds