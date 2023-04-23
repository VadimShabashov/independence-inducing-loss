import os
import csv

from datasets.dataset_utils import get_path_to_data_dir
from datasets.dataloaders_provider import DataloadersProvider


class GoogleLandmark(DataloadersProvider):
    def __init__(self, num_train_classes, num_class_samples):
        # Path to dataset
        dataset_dir = os.path.join(get_path_to_data_dir(), 'google_landmarks')

        # Get full path to images
        img_dir = os.path.join(dataset_dir, 'images')
        train_labels_path = os.path.join(dataset_dir, 'train.csv')
        test_labels_path = os.path.join(dataset_dir, 'val.csv')

        def read_csv(csv_file_path):
            data = []
            labels = []
            with open(csv_file_path, 'r') as csv_file:
                # Read csv
                csv_content = csv.reader(csv_file, delimiter=',')

                # Skip header
                next(csv_content)

                # Save info for each image
                for img_name, img_label in csv_content:
                    data.append(os.path.join(img_dir, img_name))
                    labels.append(int(img_label))

            return data, labels

        # Read info for train and test images
        train_paths, train_labels = read_csv(train_labels_path)
        test_paths, test_labels = read_csv(test_labels_path)

        super().__init__(train_paths, train_labels, test_paths, test_labels, num_train_classes, num_class_samples)
