import tensorflow as tf
import numpy as np
from  config import NO_OF_EEG_CHANNELS
from  logger import get_logger
logger = get_logger(__name__) # Logger for this module

def process_and_segment(raw_data: np.ndarray, window_size: int):
    """Processes raw EEG data by segmenting into windows, selecting channels, normalizing, 
    and reshaping for input into a CNN model."""

    num_windows = raw_data.shape[0] // window_size # # full windows we can extract from the data
    if num_windows == 0: # If the data is too short to form even one full window
        logger.warning("No full windows available")
        return None, None
    
    logger.debug(f"Number of windows: {num_windows}")

    # Only use the samples that fit into full windows (discard remainder)
    # Slicing and selecting only those that fit in full windows + reshaping
    dr = raw_data[:num_windows * window_size].reshape(num_windows, window_size, -1) # Reshape to (Windows, Time, Channels)

    # Select EEG channels and the seizure label
    x = dr[:, :, 0:NO_OF_EEG_CHANNELS]
    y = dr[:, 0, -1] # Select the LAST column as the label

    # Normalization along the time axis (axis=1)for e/ chann
    x = tf.keras.utils.normalize(x, axis=1)

    # Extra dimension for Conv2D (Windows, Time, channels (18), 1) this indicates a single channel
    x = np.expand_dims(x, axis=-1)
    
    return x, y # Return processed windowed data and labels

