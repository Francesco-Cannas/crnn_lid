from keras.layers import Input, Dense, Flatten, Dropout, Conv2D, BatchNormalization, MaxPooling2D
from keras.models import Sequential
from keras.regularizers import l2

NAME = "Topcoder_CNN"

def create_model(input_shape, config, is_training=True):

    weight_decay = 0.001

    model = Sequential()

    model.add(Input(shape=input_shape))

    model.add(Conv2D(16, (7, 7), kernel_regularizer=l2(weight_decay), activation="relu", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(32, (5, 5), kernel_regularizer=l2(weight_decay), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(64, (3, 3), kernel_regularizer=l2(weight_decay), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(128, (3, 3), kernel_regularizer=l2(weight_decay), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(256, (3, 3), kernel_regularizer=l2(weight_decay), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(1024, kernel_regularizer=l2(weight_decay), activation="relu"))

    model.add(Dense(config["num_classes"], activation="softmax"))

    return model
