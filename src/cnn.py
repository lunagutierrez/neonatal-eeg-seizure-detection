import inspect
import tensorflow as tf

def get_function_name(): 
    return inspect.stack()[1][3]

def build_cnn(input_shape):
    description = get_function_name()
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(shape=input_shape),
        
        tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding="same", name="conv2d_1"),
        tf.keras.layers.BatchNormalization(name="batch_normalization_1"),
        tf.keras.layers.Activation("relu", name="activation_relu_1"),
        tf.keras.layers.MaxPooling2D((2, 2), name="max_pooling2d_1"),
        tf.keras.layers.Dropout(0.25, name="dropout_1"),
        
        tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding="same", name="conv2d_2"),
        tf.keras.layers.BatchNormalization(name="batch_normalization_2"),
        tf.keras.layers.Activation("relu", name="activation_relu_2"),
        tf.keras.layers.MaxPooling2D((2, 2), name="max_pooling2d_2"),
        tf.keras.layers.Dropout(0.25, name="dropout_2"),
        
        tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding="same", name="conv2d_3"),
        tf.keras.layers.BatchNormalization(name="batch_normalization_3"),
        tf.keras.layers.Activation("relu", name="activation_relu_3"),
        tf.keras.layers.MaxPooling2D((2, 2), name="max_pooling2d_3"),
        tf.keras.layers.Dropout(0.5, name="dropout_3"),
        
        tf.keras.layers.Flatten(name="flatten"),
        tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001), name="dense_1"),
        tf.keras.layers.Dropout(0.5, name="dropout_4"),
        tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001), name="dense_2"),
        tf.keras.layers.Dense(1, activation="sigmoid", name="dense_3"),
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.5, nesterov=False),
        loss="binary_crossentropy",
        metrics=["acc"]
    )
    
    return model, description