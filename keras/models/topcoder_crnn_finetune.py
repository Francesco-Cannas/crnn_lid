from keras.layers import Input, Dense, Permute, Reshape, Conv2D, BatchNormalization, MaxPooling2D, Bidirectional, LSTM
from keras.models import Sequential, load_model
from keras.regularizers import l2

NAME = "Topcoder_CRNN_Finetune"

def create_model(input_shape, config):

    weight_decay = 0.001

    model = Sequential()

    model.add(Input(shape=input_shape))

    model.add(Conv2D(16, (7, 7), kernel_regularizer=l2(weight_decay), activation="relu"))
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
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 1)))

    model.add(Conv2D(256, (3, 3), kernel_regularizer=l2(weight_decay), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 1)))

    # for ref_layer in ref_model.layers:
    #     layer = model.get_layer(ref_layer.name)
    #     if layer:
    #         layer.set_weights(ref_layer.get_weights())

    for layer in model.layers:
        layer.trainable = False

    # (bs, y, x, c) --> (bs, x, y, c)
    model.add(Permute((2, 1, 3)))

    # (bs, x, y, c) --> (bs, x, y * c)
    bs, x, y, c = model.layers[-1].output.shape
    model.add(Reshape((x, y*c)))

    model.add(Bidirectional(LSTM(512, return_sequences=False), merge_mode="concat"))
    model.add(Dense(config["num_classes"], activation="softmax"))

    return model