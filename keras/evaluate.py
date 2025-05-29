import io
import sys
import argparse
import numpy as np
import data_loaders
import time
import matplotlib.pyplot as plt
import seaborn as sns
import os
from fpdf import FPDF
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
    conf_matrix = confusion_matrix(y_true, y_pred, labels=available_labels)

    output = io.StringIO()
    sys.stdout = output

    print("Accuracy: %s" % accuracy_score(y_true, y_pred))
    print("Equal Error Rate (avg): %s" % equal_error_rate(y_true, probabilities))
    print(classification_report(y_true, y_pred, labels=available_labels, target_names=label_names))
    print("Confusion Matrix:")
    print(conf_matrix)

    sys.stdout = sys.__stdout__
    report_text = output.getvalue()
    output.close()

    plot_confusion_matrix(conf_matrix, labels=label_names)
    plot_roc_curves(y_true, probabilities, label_names)

    generate_pdf_report(
        report_text,
        image_path="confusion_matrix.png",
        roc_path="roc_curves.png"
    )

def plot_confusion_matrix(conf_matrix, labels, filename="confusion_matrix.png"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_roc_curves(y_true, probabilities, label_names, filename="roc_curves.png"):
    from sklearn.metrics import roc_curve, auc

    y_true_bin = to_categorical(y_true, num_classes=len(label_names))

    plt.figure(figsize=(10, 8))
    for i in range(len(label_names)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], probabilities[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{label_names[i]} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve per Classe")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


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

   
    batches = list(data_generator.get_data(should_shuffle=False, is_prediction=True, return_labels=True))
    total_batches = len(batches)
    
    start_time = time.time()
    pbar = tqdm(batches, total=total_batches, desc="Evaluating")

    y_preds = []
    y_true_list = []
    probabilities_list = []

    for i, (data_batch, label_batch) in enumerate(pbar, 1):

        if True:
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

def generate_pdf_report(report_text, image_path=None, roc_path=None, filename="evaluation_report.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Courier", size=10)

    for line in report_text.split('\n'):
        pdf.cell(0, 10, txt=line, ln=True)

    if image_path and os.path.exists(image_path):
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Confusion Matrix", ln=True)
        pdf.image(image_path, x=10, y=30, w=180)

    if roc_path and os.path.exists(roc_path):
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "ROC Curves", ln=True)
        pdf.image(roc_path, x=10, y=30, w=180)

    pdf.output(filename)
    print(f"\nPDF report saved as: {filename}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model_dir', required=True)
    parser.add_argument('--config', dest='config', required=True)
    parser.add_argument('--testset', dest='use_test_set', default=False, action='store_true')
    cli_args = parser.parse_args()

    evaluate(cli_args)