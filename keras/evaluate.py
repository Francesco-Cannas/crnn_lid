import argparse
import numpy as np
import data_loaders
import time
from yaml import safe_load  
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from keras.models import load_model
from keras.utils import to_categorical

def equal_error_rate(y_true, probabilities):

    y_one_hot = to_categorical(y_true)
    fpr, tpr, thresholds = roc_curve(y_one_hot.ravel(), probabilities.ravel())
    eer = brentq(lambda x : 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)

    return eer


def metrics_report(y_true, y_pred, probabilities, label_names=None):

    available_labels = range(0, len(label_names))

    print("Accuracy %s" % accuracy_score(y_true, y_pred))
    print("Equal Error Rate (avg) %s" % equal_error_rate(y_true, probabilities))
    print(classification_report(y_true, y_pred, labels=available_labels, target_names=label_names))
    print(confusion_matrix(y_true, y_pred, labels=available_labels))


def evaluate(cli_args):
    import tensorflow.keras.backend as K

    def f1_score(y_true, y_pred):
        y_true = K.cast(y_true, 'float32')
        y_pred = K.cast(y_pred, 'float32')
        y_pred = K.round(y_pred)

        tp = K.sum(y_true * y_pred, axis=0)
        fp = K.sum((1 - y_true) * y_pred, axis=0)
        fn = K.sum(y_true * (1 - y_pred), axis=0)

        precision = tp / (tp + fp + K.epsilon())
        recall = tp / (tp + fn + K.epsilon())
        f1 = 2 * precision * recall / (precision + recall + K.epsilon())
        return K.mean(f1)

    with open(cli_args.config, "r") as f:
        config = safe_load(f)

    # Load Data + Labels
    dataset_dir = config["test_data_dir"] if cli_args.use_test_set else config["validation_data_dir"]

    DataLoader = getattr(data_loaders, config["data_loader"])
    data_generator = DataLoader(dataset_dir, config)

    # Model Generation
    model = load_model(cli_args.model_dir, custom_objects={'f1_score': f1_score})
    model.summary()

    total_batches = data_generator.get_num_batches()
    batches_generator = data_generator.get_data(should_shuffle=False, is_prediction=True, return_labels=True)

    start_time = time.time()
    pbar = tqdm(batches_generator, total=total_batches, desc="Evaluating")

    y_preds = []
    y_true_list = []
    probabilities_list = []

    for i, (data_batch, label_batch) in enumerate(pbar, 1):

        if i % 50 == 0 or i == 1:
            print(f"Batch {i}/{total_batches} - Input shape: {data_batch.shape} - Labels shape: {label_batch.shape}")
        
        probs_batch = model.predict(data_batch, batch_size=config["batch_size"], verbose=0)
        y_pred_batch = np.argmax(probs_batch, axis=1)
        y_true_batch = np.argmax(label_batch, axis=1)

        y_preds.append(y_pred_batch)
        y_true_list.append(y_true_batch)
        probabilities_list.append(probs_batch)

    y_preds = np.concatenate(y_preds)
    y_true_list = np.concatenate(y_true_list)
    probabilities = np.vstack(probabilities_list)

    metrics_report(y_true_list, y_preds, probabilities, label_names=config["label_names"])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model_dir', required=True)
    parser.add_argument('--config', dest='config', required=True)
    parser.add_argument('--testset', dest='use_test_set', default=False, action='store_true')
    cli_args = parser.parse_args()

    evaluate(cli_args)