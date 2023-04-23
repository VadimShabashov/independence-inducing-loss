import os
import json

from datasets.dataset_utils import get_path_to_data_dir
from datasets.dataloaders_provider import DataloadersProvider


class Oxford5k(DataloadersProvider):
    def __init__(self, num_train_classes, num_class_samples):
        # Path to dataset
        dataset_dir = os.path.join(get_path_to_data_dir(), 'oxford5k')

        # Get full paths
        img_dir = os.path.join(dataset_dir, 'images')
        groundtruth_path = os.path.join(dataset_dir, 'groundtruth.json')

        # Read groundtruth
        with open(groundtruth_path, 'r') as groundtruth_file:
            groundtruth = json.load(groundtruth_file)

        train_paths = []
        train_labels = []
        test_paths = []
        test_labels = []

        # Collect all images, that were mentioned in groundtruth file
        added_imgs = set()
        for landmark_ind, landmark in enumerate(groundtruth):
            for img_status in ['ok', 'good']:
                for img_name in groundtruth[landmark][img_status]:
                    # For some reason the same image appears for two labels. So, this 'if' helps to eliminate duplicates
                    if img_name not in added_imgs:
                        # Get and save image path
                        img_path = os.path.join(img_dir, img_name)
                        train_paths.append(img_path)

                        # Save label
                        train_labels.append(landmark_ind + 1)  # Label 0 is reserved for junk images

                        # Mark image as processed
                        added_imgs.add(img_name)

            for img_name in groundtruth[landmark]["query"]:
                # Get and save image path
                img_path = os.path.join(img_dir, img_name)
                test_paths.append(img_path)

                # Save label
                test_labels.append(landmark_ind + 1)  # Label 0 is resered for junk images

                # Save image to set to know later it has been added already
                added_imgs.add(img_path)

        # Add all images, that weren't mentioned in groundtruth (they all are junk images with label=0)
        for img_name in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_name)
            if img_name not in added_imgs:
                train_paths.append(img_path)
                train_labels.append(0)

        super().__init__(train_paths, train_labels, test_paths, test_labels, num_train_classes, num_class_samples)
