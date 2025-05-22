import argparse
import yaml
import os, time, pickle
from math import ceil, sqrt
import numpy as np
import imageio
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, MaxPooling2D, Permute, Reshape, Bidirectional, LSTM, Dense)
from tensorflow.keras.regularizers import l2
from models.topcoder_crnn_finetune import create_model

output_dir = "/mnt/c/Users/fraca/Documents/GitHub/crnn-lid/img_conv_filter"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def deprocess_image(x):
    x = x.copy()
    x -= x.mean()
    std = x.std()
    if std < 1e-5:
        return np.zeros_like(x, dtype=np.uint8)
    x /= std
    x = np.tanh(x)  
    x = (x + 1) / 2
    x = np.clip(x, 0, 1)
    x *= 255
    if K.image_data_format() == "channels_first":
        x = x.transpose((1, 2, 0))
    return x.astype("uint8")

def normalize(x):
    # Normalize tensor by its L2 norm.
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def create_stitched_image(kept_filters, img_width, img_height, margin=5):
    n = int(ceil(sqrt(len(kept_filters))))
    remaining = n * n - len(kept_filters)
    black_img = np.zeros((img_height, img_width, 1), dtype=np.uint8)
    kept_filters += [(black_img, 0.0)] * remaining

    stitched_height = n * img_height + (n - 1) * margin
    stitched_width = n * img_width + (n - 1) * margin
    stitched_filters = np.zeros((stitched_height, stitched_width, 1), dtype=np.uint8)

    for i in range(n):
        for j in range(n):
            img, loss = kept_filters[i * n + j]

            if img.ndim == 3:
                if img.shape[0] == 1 or img.shape[0] == 3: 
                    img = np.transpose(img, (1, 2, 0))
            elif img.ndim == 2:
                img = img[:, :, np.newaxis]

            if img.shape != (img_height, img_width, 1):
                resized_img = np.zeros((img_height, img_width, 1), dtype=np.uint8)
                h = min(img.shape[0], img_height)
                w = min(img.shape[1], img_width)
                resized_img[:h, :w, :] = img[:h, :w, :]
                img = resized_img

            y_start = i * (img_height + margin)
            y_end = y_start + img_height
            x_start = j * (img_width + margin)
            x_end = x_start + img_width
            stitched_filters[y_start:y_end, x_start:x_end, :] = img

    return stitched_filters

def compute_loss_and_grads(input_img_data_tensor, model_input, layer_output, filter_index):
    with tf.GradientTape() as tape:
        tape.watch(input_img_data_tensor)
        activation = model_input(input_img_data_tensor)
        loss = tf.reduce_mean(layer_output[:, :, :, filter_index])
    grads = tape.gradient(loss, input_img_data_tensor)
    grads = grads / (tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)
    return loss, grads

def visualize_conv_filters(model, layer, num_filters, img_width, img_height,
                           max_filters=64, gradient_steps=20, step_size=1.0, save_pickle=False, load_pickle=False):
    layer_name = layer.name
    pickle_file = f"{layer_name}_filters.pickle"

    if load_pickle:
        try:
            kept_filters = pickle.load(open(pickle_file, "rb"))
            print(f"Loaded filters from {pickle_file}")
        except Exception as e:
            print(f"Could not load filters pickle: {e}")
            load_pickle = False  # fallback to compute filters

    if not load_pickle:
        kept_filters = []
        intermediate = Model(inputs=model.input, outputs=layer.output)

        for filter_index in range(min(num_filters, max_filters)):
            print(f"Processing filter {filter_index} in layer {layer_name}")
            start_time = time.time()

            if K.image_data_format() == "channels_first":
                input_shape = (1, 1, img_height, img_width)
            else:
                input_shape = (1, img_height, img_width, 1)

            input_img_data = np.random.uniform(low=-0.5, high=0.5, size=input_shape).astype(np.float32)
            input_img_data_tensor = tf.Variable(input_img_data)

            for step in range(gradient_steps):
                with tf.GradientTape() as tape:
                    tape.watch(input_img_data_tensor)
                    activation = intermediate(input_img_data_tensor)
                    loss = tf.reduce_mean(activation[:, :, :, filter_index])

                grads = tape.gradient(loss, input_img_data_tensor)
                grads = grads / (tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)
                input_img_data_tensor.assign_add(grads * step_size)

                if loss <= 0:
                    print(f"Filter {filter_index} got stuck at step {step} with loss {loss.numpy()}")
                    break

            if loss > 0:
                img = deprocess_image(input_img_data_tensor.numpy()[0])
                kept_filters.append((img, loss.numpy()))
            else:
                print(f"Skipping filter {filter_index} due to non-positive loss")

            end_time = time.time()
            print(f"Filter {filter_index} processed in {int(end_time - start_time)}s")

        if save_pickle:
            pickle.dump(kept_filters, open(pickle_file, "wb"))
            print(f"Saved filters to {pickle_file}")

    kept_filters.sort(key=lambda x: x[1], reverse=True)
    kept_filters = kept_filters[:max_filters]

    stitched_filters = create_stitched_image(kept_filters, img_width, img_height)

    output_img_name = f"{layer.name}_{int(sqrt(max_filters))}x{int(sqrt(max_filters))}.png"
    output_path = os.path.join(output_dir, output_img_name)
    imageio.imwrite(output_path, np.squeeze(stitched_filters))
    print(f"Saved stitched image to {output_img_name}")

def visualize_conv_layers(cli_args, config):
    img_width = cli_args.width if cli_args.width else config["input_shape"][1]
    img_height = cli_args.height if cli_args.height else config["input_shape"][0]

    input_shape = tuple(config["input_shape"])
    model = create_model(input_shape, config)

    dummy_input = np.random.randn(1, *input_shape).astype(np.float32)
    model(dummy_input)

    model.compile(optimizer="adam", loss="categorical_crossentropy")
    model.summary()

    for layer in model.layers:
        if "conv" in layer.name and hasattr(layer, "output") and layer.output is not None:
            try:
                intermediate_model = tf.keras.Model(inputs=model.input, outputs=layer.output)
                num_filters = layer.output.shape[-1]
                print(f"Visualizing layer {layer.name} with {num_filters} filters")
                visualize_conv_filters(model, layer, num_filters,
                                    img_width, img_height, max_filters=cli_args.num_filter,
                                    gradient_steps=20, step_size=1.0,
                                    save_pickle=cli_args.save_pickle, load_pickle=cli_args.load_pickle)
            except Exception as e:
                print(f"Skipping layer {layer.name} due to error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True)
    parser.add_argument("--layer", dest="layer_name", default="convolution2d_1")
    parser.add_argument('--width', dest='width', type=int)
    parser.add_argument('--height', dest='height', type=int)
    parser.add_argument('--filter', dest='num_filter', default=64, type=int)
    parser.add_argument('--save_pickle', dest='save_pickle', default=False, action='store_true')
    parser.add_argument('--load_pickle', dest='load_pickle', default=False, action='store_true')
    cli_args = parser.parse_args()

    config_path = os.path.abspath(cli_args.config)
    with open(config_path, "rb") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    visualize_conv_layers(cli_args, config)