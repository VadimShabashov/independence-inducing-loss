import os
import numpy as np
from sklearn.model_selection import train_test_split

from datasets.dataset_utils import get_path_to_data_dir
from datasets.dataloaders_provider import DataloadersProvider


class CIFAR10(DataloadersProvider):
    def __init__(self, num_train_classes, num_class_samples):
        # Path to dataset
        dataset_dir = os.path.join(get_path_to_data_dir(), 'cifar10')

        # Define correspondence between folder name and label
        folder_to_label = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6,
                           'horse': 7, 'ship': 8, 'truck': 9}

        # Define paths and labels of all images
        imgs_paths = []
        imgs_labels = []

        def get_imgs_from_dir(img_dir):
            for folder_name in os.listdir(img_dir):
                folder_path = os.path.join(img_dir, folder_name)
                for img_name in os.listdir(folder_path):
                    # Image full path
                    img_path = os.path.join(folder_path, img_name)

                    # Save image path and label
                    imgs_paths.append(img_path)
                    imgs_labels.append(folder_to_label[folder_name])

        # Get full paths to train and test folders
        img_dir_train = os.path.join(dataset_dir, 'train')
        img_dir_test = os.path.join(dataset_dir, 'test')

        # Collect all images paths
        get_imgs_from_dir(img_dir_train)
        get_imgs_from_dir(img_dir_test)

        # Make train/test split. Test will consist of 100 samples equally sampled from each of the classes
        train_idx, test_idx = train_test_split(
            np.arange(len(imgs_labels)),
            test_size=100,
            shuffle=True,
            stratify=imgs_labels
        )

        # For quick access by index
        imgs_paths = np.array(imgs_paths)
        imgs_labels = np.array(imgs_labels)

        super().__init__(
            imgs_paths[train_idx], imgs_labels[train_idx], imgs_paths[test_idx], imgs_labels[test_idx],
            num_train_classes, num_class_samples
        )
