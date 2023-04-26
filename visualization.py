import os
import csv
import click
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable


def plot_metric_histograms(experiment_path, experiments, metrics):
    # Get number of experiments and histograms
    num_experiments = len(experiments)
    num_histograms = len(metrics)

    # Get figure
    fig, axs = plt.subplots(num_experiments, num_histograms + 1,
                            figsize=(5 * (num_histograms + 1), 5 * num_experiments))

    for experiment_ind, experiment in enumerate(experiments):
        # Get experiment parameters
        dataset_name, embedding_dim, regularization_loss, classification_loss, batch, margin, model_name, \
            independence_loss, num_epochs_passed, metrics_storage = experiment

        # Get experiment axis
        if num_experiments == 1:
            exp_ax = axs
        else:
            exp_ax = axs[experiment_ind]

        # Add histograms to the plot
        for metric_ind, metric in enumerate(metrics):
            # Get values for metric
            metric_values = metrics_storage.get_metric(metric)

            # Select few samples from distribution (to avoid too long hist calculation)
            num_samples = min(len(metric_values), 1000)

            # Choose number of bins
            if metric == "P@1":
                bins = 2
            else:
                bins = 10

            # Add histogram to the axis
            exp_ax[metric_ind].hist(np.random.choice(metric_values, size=num_samples, replace=False), bins=bins)
            exp_ax[metric_ind].set_title(metric)

        # Create array, containing descriptions
        experiment_config = [
            f"Dataset = {dataset_name}",
            f"Emb. dim = {embedding_dim}",
            f"Reg. loss = {regularization_loss}",
            f"Class. loss = {classification_loss}",
            f"Batch = {batch}",
            f"Margin = {margin}",
            f"Model = {model_name}",
            f"Ind. loss = {independence_loss}",
            f"Epochs = {num_epochs_passed}"
        ]

        # Add description as the last subplot
        exp_ax[num_histograms].text(
            0.1, 0.5, "\n".join(experiment_config),
            horizontalalignment='left', verticalalignment='center',
            transform=exp_ax[num_histograms].transAxes, fontsize=11
        )
        exp_ax[num_histograms].axis('off')

    # Save histograms
    histograms_path = os.path.join(experiment_path, 'histograms.png')
    plt.savefig(histograms_path)


@click.command()
@click.option('--experiment_path', help='Path to the folder with experiments results')
def visualize_results(experiment_path):
    # Get path to the csv file with results
    csv_results_path = os.path.join(experiment_path, 'results.csv')

    # Create table csv file visualization
    results_table = PrettyTable()

    with open(csv_results_path, 'r') as csv_results_file:
        # Read csv
        csv_content = csv.reader(csv_results_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)

        # Add header
        results_table.field_names = next(csv_content)

        # Add configuration and result for each experiment (replace None with '-')
        for results_row in csv_content:
            results_table.add_row(
                list(map(lambda field: field if field else '-', results_row))
            )

    # Show results table with metrics up to 2nd digit
    print("Results table:")
    results_table.float_format = '.2'
    print(results_table)


if __name__ == '__main__':
    visualize_results()
