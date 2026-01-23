import tensorflow as tf
import numpy as np
from config import NO_OF_EEG_CHANNELS
from logger import get_logger
logger = get_logger(__name__)

def process_and_segment(raw_data: np.ndarray, window_size: int):
    """Reshapes data into windows, selects channels, and normalizes."""
    num_windows = raw_data.shape[0] // window_size
    if num_windows == 0:
        logger.warning("No full windows available")
        return None, None
    
    logger.debug(f"Number of windows: {num_windows}")
    
    # Reshape to (Windows, Time, Channels)
    dr = raw_data[:num_windows * window_size].reshape(num_windows, window_size, -1)

    # Select 18 EEG channels and the seizure label
    x = dr[:, :, 0:NO_OF_EEG_CHANNELS]
    y = dr[:, 0, -1] #Select the LAST column as the label

    # Normalization along the time axis (axis=1)
    x = tf.keras.utils.normalize(x, axis=1)

    # Extra dimension for Conv2D (Windows, Time, 18, 1)
    x = np.expand_dims(x, axis=-1)
    
    return x, y

