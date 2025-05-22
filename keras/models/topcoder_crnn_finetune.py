from keras.layers import Input, Dense, Permute, Reshape, Conv2D, BatchNormalization, MaxPooling2D, Bidirectional, LSTM
from keras.models import Model
from keras.regularizers import l2

NAME = "Topcoder_CRNN_Finetune"

def create_model(input_shape, config):

    weight_decay = 0.001
    inputs = Input(shape=input_shape)

    x = Conv2D(16, (7, 7), kernel_regularizer=l2(weight_decay), activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(32, (5, 5), kernel_regularizer=l2(weight_decay), activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3), kernel_regularizer=l2(weight_decay), activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), kernel_regularizer=l2(weight_decay), activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 1))(x)

    x = Conv2D(256, (3, 3), kernel_regularizer=l2(weight_decay), activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 1))(x)

    # for ref_layer in ref_model.layers:
    #     layer = model.get_layer(ref_layer.name)
    #     if layer:
    #         layer.set_weights(ref_layer.get_weights())

    # LSTM + Dense
    x = Permute((2, 1, 3))(x)  # (bs, x, y, c)
    x_shape = x.shape
    x = Reshape((x_shape[1], x_shape[2] * x_shape[3]))(x)  # (bs, x, y*c)

    x = Bidirectional(LSTM(512, return_sequences=False))(x)
    outputs = Dense(config["num_classes"], activation="softmax")(x)
    
    model = Model(inputs=inputs, outputs=outputs)

    for layer in model.layers:
        if isinstance(layer, (Conv2D, BatchNormalization, MaxPooling2D)):
            layer.trainable = False

    return model