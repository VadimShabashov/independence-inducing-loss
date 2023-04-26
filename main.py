import os
import csv
import torch
import click
import logging
import pytorch_lightning as pl

from model import Model, run_inference
from metrics_storage import MetricsStorage
from utils import get_dataset, get_independence_loss, get_regularization_loss
from config_parser import parse_config
from visualization import plot_metric_histograms


import ssl
ssl._create_default_https_context = ssl._create_unverified_context


@click.command()
@click.option('--experiment_path', help='Path to the folder with experiment config')
def main(experiment_path):
    # Path to the log file, where training status is saved
    log_path = os.path.join(experiment_path, 'experiment.log')

    # Create logger
    logging.basicConfig(level=logging.INFO, filename=log_path, filemode='w', format='%(message)s')
    logger = logging.getLogger()

    try:
        # Get experiments parameters
        config = parse_config(experiment_path)

        # Get metrics to track, metrics to visualize on histogram and their union
        track_metrics = config['track_metric']
        plot_hist_metrics = config['plot_hist_metric']
        all_metrics = set(track_metrics).union(set(plot_hist_metrics))

        # Get info about number of epochs
        num_epochs_in_step = config['num_epochs_in_step']
        num_epoch_steps = config['num_epoch_steps']

        # Get device (GPU if it's available)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device='{device}'")

        # For tracking metrics and saving to csv in the end
        tracking_results = []
        fields = [
            'Dataset', 'Emb. dim', 'Reg. loss', 'Class. loss', 'Batch', 'Margin', 'Model', 'Ind. loss', 'Epochs',
            *track_metrics
        ]

        # For histograms visualization
        experiments = []

        for dataset_name in config['dataset']:
            for batch in config['batch']:

                # Get info about batch
                num_train_classes, num_class_samples = tuple(map(int, batch.split('/')))

                # Get dataset
                dataset = get_dataset(dataset_name, num_train_classes, num_class_samples)

                # Get dataloaders and number of classes for classification
                train_dataloader = dataset.get_train_dataloader()
                test_database_dataloader, test_query_dataloader = dataset.get_test_dataloaders()
                num_classification_classes = dataset.get_number_of_classes()

                for embedding_dim in config['embedding_dim']:
                    for regularization_loss_name in config['regularization_loss']:
                        # Get regularization loss
                        regularization_loss = get_regularization_loss(regularization_loss_name, device)

                        for classification_loss in config['classification_loss']:
                            for margin in config['margin']:
                                for model_name in config['model']:
                                    for independence_loss_name in config['independence_loss']:
                                        # Get independence loss
                                        independence_loss = get_independence_loss(independence_loss_name, device)

                                        # Display current configuration, so that we can track train losses
                                        logger.info(
                                            "\nModel with configuration: " +
                                            f"dataset={dataset_name}, model={model_name}, embedding dim={embedding_dim}, " +
                                            f"ind. loss={independence_loss_name}, reg. loss={regularization_loss_name}, " +
                                            f"class. loss={classification_loss}, batch={batch}, " +
                                            f"margin={margin}, num_epochs_in_step={num_epochs_in_step}, " +
                                            f"num_epoch_steps={num_epoch_steps}"
                                        )

                                        # Create model
                                        model = Model(
                                            model_name, device, independence_loss, regularization_loss,
                                            classification_loss, num_classification_classes, embedding_dim, margin
                                        )

                                        for epoch_step in range(num_epoch_steps):
                                            # Create trainer
                                            trainer = pl.Trainer(
                                                max_epochs=num_epochs_in_step,
                                                accelerator=device
                                            )

                                            # Run training
                                            trainer.fit(model, train_dataloader)

                                            # Get database and query embeddings
                                            database_embeddings, database_labels = run_inference(
                                                trainer, test_database_dataloader, model
                                            )
                                            query_embeddings, query_labels = run_inference(
                                                trainer, test_query_dataloader, model
                                            )

                                            # Calculate metrics
                                            metrics_storage = MetricsStorage(
                                                all_metrics,
                                                database_embeddings, database_labels,
                                                query_embeddings, query_labels
                                            )

                                            # Calculate mean of track metrics
                                            mean_track_metrics = metrics_storage.get_mean_of_metrics(track_metrics)

                                            # Calculate overall
                                            num_epochs_passed = num_epochs_in_step * (epoch_step + 1)

                                            # Add results to table
                                            tracking_results.append([
                                                dataset_name, embedding_dim,
                                                regularization_loss_name, classification_loss,
                                                batch, margin, model_name,
                                                independence_loss_name, num_epochs_passed,
                                                *mean_track_metrics
                                            ])

                                            # Save experiment configuration and calculated metrics
                                            if plot_hist_metrics:
                                                experiments.append(
                                                    (
                                                        dataset_name, embedding_dim,
                                                        regularization_loss_name, classification_loss,
                                                        batch, margin, model_name,
                                                        independence_loss_name, num_epochs_passed,
                                                        metrics_storage
                                                    )
                                                )

        # Save mean metrics results
        csv_results_path = os.path.join(experiment_path, 'results.csv')
        with open(csv_results_path, 'w') as csv_results_file:
            # Get writer
            write = csv.writer(csv_results_file, quoting=csv.QUOTE_NONNUMERIC)

            # Write fields and results
            write.writerow(fields)
            write.writerows(tracking_results)

        # Show histograms for metrics
        if plot_hist_metrics:
            plot_metric_histograms(experiment_path, experiments, plot_hist_metrics)

    except Exception as e:
        logging.error(f'Exception: {e}')


if __name__ == '__main__':
    main()
