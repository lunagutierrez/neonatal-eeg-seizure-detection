import inspect
import tensorflow as tf
from config import LEARNING_RATE

def get_function_name(): 
    return inspect.stack()[1][3] # Return name of the caller function as str

def build_cnn(input_shape):
    description = get_function_name() # Get the name of this function dynamically for logging/description
    
    model = tf.keras.models.Sequential([
        #Input layer
        tf.keras.layers.InputLayer(shape=input_shape),
        
        # 1st Convolutional Block
        tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding="same", name="conv2d_1"), # filters, kernel_size, stride of conv, size(out) = size(in)
        tf.keras.layers.BatchNormalization(name="batch_normalization_1"),
        tf.keras.layers.Activation("relu", name="activation_relu_1"),
        tf.keras.layers.MaxPooling2D((2, 2), name="max_pooling2d_1"), # Reduce spatial dimensions /2
        tf.keras.layers.Dropout(0.25, name="dropout_1"), # Dropout for regularization & prevent overfitting
        
        # 2nd Convolutional Block
        tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding="same", name="conv2d_2"),
        tf.keras.layers.BatchNormalization(name="batch_normalization_2"),
        tf.keras.layers.Activation("relu", name="activation_relu_2"),
        tf.keras.layers.MaxPooling2D((2, 2), name="max_pooling2d_2"),
        tf.keras.layers.Dropout(0.25, name="dropout_2"),
        
        # 3rd Convolutional Block
        tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding="same", name="conv2d_3"),
        tf.keras.layers.BatchNormalization(name="batch_normalization_3"),
        tf.keras.layers.Activation("relu", name="activation_relu_3"),
        tf.keras.layers.MaxPooling2D((2, 2), name="max_pooling2d_3"),
        tf.keras.layers.Dropout(0.5, name="dropout_3"),
        
        # Flatten + Fully Connected Layers
        tf.keras.layers.Flatten(name="flatten"), # Flatten feature maps to 1D feature vector
        tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001), name="dense_1"), # L2 reg to prevent overfitting
        tf.keras.layers.Dropout(0.5, name="dropout_4"),
        tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001), name="dense_2"),
        tf.keras.layers.Dense(1, activation="sigmoid", name="dense_3"), # Binary classification output layer
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.5, nesterov=False), #momentum for smoother updates, not using Nesterov acceleration
        loss="binary_crossentropy",
        metrics=["acc"]
    )
    
    return model, description