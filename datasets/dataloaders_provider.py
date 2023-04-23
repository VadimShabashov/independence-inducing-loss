from torch.utils.data import DataLoader

from datasets.base_dataset import BaseDataset
from datasets.triplet_batch_sampler import TripletBatchSampler


class DataloadersProvider:
    def __init__(self, train_paths, train_labels, test_paths, test_labels, num_train_classes, num_class_samples):
        # Number of workers for dataloaders
        num_workers = 4

        # Datasets for train and test (test consists of query images and database to look in)
        train_dataset = BaseDataset(train_paths, train_labels, training=True)
        test_database_dataset = BaseDataset(train_paths, train_labels, training=False)
        test_query_dataset = BaseDataset(test_paths, test_labels, training=False)

        # Defining batch sampler for training
        train_batch_sampler = TripletBatchSampler(train_dataset.labels, num_train_classes, num_class_samples)

        # Dataloader for train and test
        self.train_dataloader = DataLoader(
            train_dataset, batch_sampler=train_batch_sampler, num_workers=num_workers
        )
        self.test_database_dataloader = DataLoader(
            test_database_dataset, batch_size=64, shuffle=False, num_workers=num_workers
        )
        self.test_query_dataloader = DataLoader(
            test_query_dataset, batch_size=64, shuffle=False, num_workers=num_workers
        )

        # Save number of classes
        self.num_classes_overall = max(train_labels) + 1

    def get_train_dataloader(self):
        return self.train_dataloader

    def get_test_dataloaders(self):
        return self.test_database_dataloader, self.test_query_dataloader

    def get_number_of_classes(self):
        return self.num_classes_overall
