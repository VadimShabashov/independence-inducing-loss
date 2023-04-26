import random
from collections import defaultdict


class TripletBatchSampler:
    """
    Batch sampler, that randomly selects num_classes, then randomly select min(num_samples, size of class) from them.
    The idea of such sampling technique: https://arxiv.org/pdf/1703.07737.pdf
    Similar implementation: https://github.com/adambielski/siamese-triplet/blob/master/datasets.py
    """

    def __init__(self, labels, num_classes, num_samples):
        self.labels = labels
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.batch_size = num_classes * num_samples

        # Create dict: label=[indices with such label]
        self.label_to_indices = defaultdict(list)
        for label_ind, label in enumerate(labels):
            self.label_to_indices[label].append(label_ind)

        # Find distinct labels (classes)
        self.distinct_labels = list(self.label_to_indices.keys())

    def __iter__(self):
        num_used_images = 0
        while num_used_images < len(self.labels):
            # Choose classes randomly
            classes = random.sample(self.distinct_labels, self.num_classes)

            # Get indices for each of the class
            indices = []
            for class_ in classes:
                # Get number of samples, allowed for this class
                class_num_samples = min(len(self.label_to_indices[class_]), self.num_samples)

                # Randomly sample indices
                class_indices = random.sample(self.label_to_indices[class_], class_num_samples)

                # Update number of used images and overall indices list
                num_used_images += len(class_indices)
                indices.extend(class_indices)

            yield indices

    def __len__(self):
        return len(self.labels) // (self.num_classes * self.num_samples)
