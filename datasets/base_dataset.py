from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, img_paths, labels, training):
        super().__init__()

        self.img_paths = img_paths
        self.labels = labels
        self.training = training

    @staticmethod
    def get_img(img_path, training):
        """
        Function, that reads image, transforms according to training flag, then returns it
        """

        # Size of image, supported by all models
        resize_size = 224

        # Open image and find its minimum size
        img = Image.open(img_path)
        min_size = min(img.size[0], img.size[1])

        # Define transforms (image is cut to a square then resized)
        if training:
            transforms = T.Compose([
                T.ToTensor(),
                T.RandomHorizontalFlip(0.5),
                T.RandomCrop(min_size),
                T.Resize(resize_size, antialias=False)
            ])
        else:
            transforms = T.Compose([
                T.ToTensor(),
                T.CenterCrop(min_size),
                T.Resize(resize_size, antialias=False)
            ])

        return transforms(img)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.get_img(self.img_paths[idx], self.training), self.labels[idx]
