import argparse
import numpy as np
from yaml import safe_load  
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from keras.models import load_model
from keras.utils import to_categorical
import data_loaders

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

    with open(cli_args.config, "r") as f:
        config = safe_load(f)

    # Load Data + Labels
    dataset_dir = config["test_data_dir"] if cli_args.use_test_set else config["validation_data_dir"]

    DataLoader = getattr(data_loaders, config["data_loader"])
    data_generator = DataLoader(dataset_dir, config)

    # Model Generation
    model = load_model(cli_args.model_dir)
    print(model.summary())

    probabilities = model.predict(
        data_generator.get_data(should_shuffle=False, is_prediction=True),
        steps=data_generator.get_num_files()
    )

    y_pred = np.argmax(probabilities, axis=1)
    y_true = data_generator.get_labels()[:len(y_pred)]
    metrics_report(y_true, y_pred, probabilities, label_names=config["label_names"])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model_dir', required=True)
    parser.add_argument('--config', dest='config', required=True)
    parser.add_argument('--testset', dest='use_test_set', default=False, action='store_true')
    cli_args = parser.parse_args()

    evaluate(cli_args)