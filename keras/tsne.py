import argparse
import os
import math
import numpy as np
import yaml
import data_loaders
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
from sklearn.manifold import TSNE
from pandas import DataFrame
from models.topcoder_crnn_finetune import create_model

def plot_with_labels(lowD_Weights, labels, label_names, filename):
    df = DataFrame({"x": lowD_Weights[:, 0], "y": lowD_Weights[:, 1], "label": labels})
    groups = df.groupby("label")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.margins(0.05)

    unique_labels = sorted(set(labels))
    colors = plt.get_cmap('tab10', len(unique_labels))

    for idx, label in enumerate(unique_labels):
        group = groups.get_group(label)
        ax.scatter(group.x, group.y, 
                   color=colors(idx), 
                   label=label_names[label], 
                   alpha=0.7, 
                   s=60, 
                   edgecolors='k', 
                   linewidth=0.5)

        centroid_x = group.x.mean()
        centroid_y = group.y.mean()
        ax.text(centroid_x, centroid_y, label_names[label], fontsize=12, weight='bold',
                horizontalalignment='center', verticalalignment='center', 
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    ax.legend(title="Classes", loc='best', fontsize=10)
    ax.set_title('t-SNE visualization of model outputs', fontsize=16)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')

    fig.tight_layout()
    fig.savefig(filename)
    fig.savefig(filename.replace('.pdf', '.png'))

    plt.close(fig)

    df = DataFrame({"x": lowD_Weights[:, 0], "y": lowD_Weights[:, 1], "label": labels})
    groups = df.groupby("label")

    fig, ax = plt.subplots()
    ax.margins(0.05)
    for label, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=6, label=label_names[label])
    ax.legend(numpoints=1)

    cur_axes = plt.gca()
    cur_axes.axes.get_xaxis().set_visible(False)
    cur_axes.axes.get_yaxis().set_visible(False)
    cur_axes.axes.get_xaxis().set_ticks([])
    cur_axes.axes.get_yaxis().set_ticks([])

    fig = plt.gcf()
    fig.savefig(filename)

def visualize_cluster(cli_args):

    config_path = os.path.abspath(cli_args.config)
    plot_path = os.path.abspath(cli_args.plot_name)

    config = yaml.load(open(config_path, "rb"), Loader=yaml.SafeLoader)

    DataLoader = getattr(data_loaders, config["data_loader"])
    data_generator = DataLoader(config["validation_data_dir"], config)

    input_shape = tuple(config["input_shape"])
    model = create_model(input_shape, config)
    model.compile(optimizer="adam", loss="categorical_crossentropy")

    steps = math.ceil(data_generator.get_num_files() / config["batch_size"])
    probabilities = model.predict(
        data_generator.get_data(should_shuffle=False, is_prediction=True),
        steps=steps,
        verbose=1
    )
    print(f"Probabilities shape: {probabilities.shape}")

    images_label_pairs = data_generator.images_label_pairs
    if len(images_label_pairs) != probabilities.shape[0]:
        print(f"Warning: numero di label ({len(images_label_pairs)}) e probabilità ({probabilities.shape[0]}) non corrispondono!")

    limit = cli_args.limit
    num_samples = min(probabilities.shape[0], len(images_label_pairs), limit)
    print(f"User limit: {limit} | Num samples used: {num_samples}")

    directory = os.path.dirname(plot_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    prob_path = os.path.join(directory, "probabilities.tsv")
    np.savetxt(prob_path, probabilities[:num_samples], delimiter='\t', newline='\n')

    file_names, labels = zip(*images_label_pairs[:num_samples])

    unique_labels = sorted(set(labels))
    for label in unique_labels:
        count = labels.count(label)
        label_name = config["label_names"][label] if label in config["label_names"] else str(label)
        print(f" - {label_name}: {count} esempi")

    meta_path = os.path.join(directory, "metadata.tsv")
    df = DataFrame({"label": labels, "filename": file_names})
    df.to_csv(meta_path, sep="\t", float_format="%.18e", header=True, index=False)
    print(f"Metadata salvati in: {meta_path}")

    tsne = TSNE(perplexity=30, n_components=2, init='pca', max_iter=cli_args.num_iter)
    lowD_weights = tsne.fit_transform(probabilities[:num_samples, :])

    # Crea il plot
    plot_with_labels(lowD_weights, labels, config["label_names"], plot_path)
    print(f"Plot salvato in: {plot_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True)
    parser.add_argument('--plot', dest='plot_name')
    parser.add_argument('--limit', dest='limit', default=2000, type=int)
    parser.add_argument('--iter', dest='num_iter', default=4000, type=int)
    cli_args = parser.parse_args()

    cli_args.plot_name = "tsne.pdf"

    visualize_cluster(cli_args)