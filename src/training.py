from sklearn.model_selection import KFold
from logger import get_logger

logger = get_logger(__name__)

def train_kfold(model_fn, X, y, config):
    histories = []

    kf = KFold(n_splits=config.KFOLDS, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        logger.info(f"Training fold {fold}")

        model = model_fn()

        history = model.fit(
            X[train_idx], y[train_idx],
            validation_data=(X[val_idx], y[val_idx]),
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            verbose=0
        )

        histories.append((model, history))

    return histories
