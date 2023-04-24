# Independence inducing loss

## Description

The repository contains a platform for convenient performance comparison of various independence inducing losses
for image retrieval problem.

The underlying idea behind independence inducing loss is making features in embedding to be independent. The induced
independence force all high-level image characteristics to be contained in different components, and also avoid duplicating each other.

This lead to several advantageous properties of the embeddings:
1. Sparsity.
2. More correct calculation of the similarity between embeddings.
3. Better interpretability of embeddings.

## Setting experiment

The experiment configuration is set in the `config.yml`.

Description of the parameters:

|      **Parameter**      |                                            **Description**                                             |                            **Values**                           | **Syntax example** | **Required** |
|:-----------------------:|:------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------:|:------------------:|:------------:|
|         dataset         |                                            Datasets to use.                                            |                CIFAR10, Oxford5k, GoogleLandmarks               |       CIFAR10      |     True     |
|          model          |                                             Models to use.                                             |            AlexNet, ResNet18, ResNet34, ResNet50, ViT           |      ResNet18      |     True     |
|      embedding_dim      |                                      Dimension of the embeddings.                                      |                       Any positive integer                      |        1024        |     True     |
|    independence_loss    |                                           Independence loss.                                           | null, NegApproxLoss1, NegApproxLoss2, KurtosisLoss, CorrMatLoss |   NegApproxLoss1   |     True     |
|   regularization_loss   |                                          Regularization loss.                                          |                             null, L1                            |         L1         |     True     |
| use_classification_loss |                          Boolean flag, whether to use classification or not.                           |                           true, false                           |        true        |     True     |
|          margin         |                                   Margin for ranking (triplet) loss.                                   |                         positive double                         |         0.2        |     True     |
|    num_epochs_in_step   |                                Number epochs between checking metrics.                                 |                         positive integer                        |         10         |     True     |
|     num_epoch_steps     |                                Number of steps (times) to check metrics                                |                         positive integer                        |          5         |     True     |
|          batch          | Two integers for batch sample: number of classes to select, number of samples for each of the classes. |         Pair of two positive integers, separated with /         |         5/5        |     True     |
|       track_metric      |                               Metrics to track and save in the csv file.                               |                   P@k, Independence, Sparsity                   |         P@1        |     True     |
|     plot_hist_metric    |                                    Metrics to plot histograms for.                                     |                   P@k, Independence, Sparsity                   |         P@1        |     False    |

Note: for some parameters several values can be set. In that case platform will run experiment for each of the
possible configurations.

For example, several datasets and models can be provided in the configuration file. In that case, experiments will be
carried out for each possible pair of dataset and model.

## Example of `config.yml`

```yaml
dataset:
    - CIFAR10
    - Oxford5k
model:
    - ResNet18
    - ViT
embedding_dim:
    - 1024
independence_loss:
    - null
regularization_loss:
    - null
use_classification_loss:
    - true
margin:
    - 0.5
num_epochs_in_step: 1
num_epoch_steps: 1
batch:
    - 5/5
track_metric:
    - P@1
    - P@3
    - P@5
    - P@10
    - Independence
    - Sparsity
plot_hist_metric:
    - P@1
    - P@10
    - Independence
    - Sparsity
```

## Outline for starting the platform

1. Clone/download this repo.
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```
3. Download desired datasets, extract from archives put them into `datasets/data`.
   * CIFAR10
   
      Download [link](https://www.kaggle.com/datasets/swaroopkml/cifar10-pngs-in-folders)
      
      Required project structure:
      ```
      ├── datasets
      │   ├── data
      │   │   ├── cifar10
      │   │   │   ├── train
      │   │   │   ├── test
      ```
   * Oxford5k
   
      Download [link](https://www.kaggle.com/datasets/vadimshabashov/oxford5k)
      
      Required project structure:
      ```
      ├── datasets
      │   ├── data
      │   │   ├── oxford5k
      │   │   │   ├── images
      │   │   │   ├── groundtruth.json
      ```
   * Google Landmarks (subset)
   
      Download [link](https://www.kaggle.com/datasets/confirm/google-landmark-dataset-v2-micro)
      
      Required project structure:
      ```
      ├── datasets
      │   ├── data
      │   │   ├── google_landmarks
      │   │   │   ├── images
      │   │   │   ├── train.csv
      │   │   │   ├── val.csv
      ```
4. Specify the experiment configuration in `config.yml`
5. Start platform with the command:
   ```
   python3 main.py --experiment_path=<path to the config.yml>
   ```

   Example:
   ```
   python3 main.py --experiment_path=experiments/experiment1
   ```
6. The results will be saved in the `experiment_path` near the `config.yml` file.

   * Results for `track_metrics` are saved in the `results.csv`.
   * Histograms for `plot_hist_metric` are saved in the `histograms.png`.
   
7. Optional. For convenient visualization of `track_metrics`, one can run the following script:
   ```
   python3 visualization --experiment_path=<path to the config.yml>
   ```
   
   The script reads the `results.csv` file and prints it as table to the output.
