import os.path
import re
import yaml


def validate_metrics(metrics):
    # Get all unknown metrics
    p_at_k_pattern = re.compile("P@[1-9][0-9]*$")
    unknown_metrics = []
    for metric in metrics:
        if metric != 'Independence' and metric != 'Sparsity' and metric != 'MAP' and not p_at_k_pattern.match(metric):
            unknown_metrics.append(metric)

    if unknown_metrics:
        raise Exception(f"Unknown metrics: {unknown_metrics}")


def validate_models(models):
    supported_models = ["AlexNet", "ResNet18", "ResNet34", "ResNet50", "ViT"]
    unknown_models = []
    for model in models:
        if model not in supported_models:
            unknown_models.append(model)

    if unknown_models:
        raise Exception(f"Unknown models: {unknown_models}")


def parse_config(experiment_path):
    # Get path to the experiment config
    config_path = os.path.join(experiment_path, 'config.yml')

    # Get config file
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Check that all required fields were provided
    required_fields = [
        'dataset', 'model', 'embedding_dim',
        'independence_loss', 'regularization_loss', 'classification_loss',
        'margin', 'num_epochs_in_step', 'num_epoch_steps', 'batch', 'track_metric'
    ]
    missing_fields = []
    for field in required_fields:
        if field not in config:
            missing_fields.append(field)

    if missing_fields:
        raise Exception(f"Add missing fields to config file: {missing_fields}")

    # If optional field 'plot_hist_metric' wasn't used, set default value (empty list of metrics)
    if 'plot_hist_metric' not in config:
        config['plot_hist_metric'] = []

    # Validate models and metrics
    validate_models(config['model'])
    validate_metrics(config['track_metric'])
    validate_metrics(config['plot_hist_metric'])

    return config


def get_number_experiments(config):
    multiple_values_fields = [
        'dataset', 'model', 'embedding_dim',
        'independence_loss', 'regularization_loss', 'classification_loss',
        'margin', 'batch'
    ]

    overall_number_experiments = 1
    for multiple_values_field in multiple_values_fields:
        overall_number_experiments *= len(config[multiple_values_field])

    return overall_number_experiments
