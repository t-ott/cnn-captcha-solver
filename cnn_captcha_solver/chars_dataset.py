import os
import pandas as pd
from PIL import Image
import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset

class CharsDataset(Dataset):
    '''
    Custom PyTorch Dataset to load in individual character images segmented
    from CAPTCHA images

    Parameters
    ----------
    annotations_path : str
        Path to annotations CSV file of training data, with each row
        containing (filename, label)
    root_dir : str
        Path to directory containing segmented character training images

    Attributes
    ----------
    annotations : pandas.DataFrame
        Pandas DataFrame with character image filenames (column 0, named
        'filenames') and their corresponding integer labels (column 1, named
        'label')
    root_dir : str
        Path to the directory which contains the character training images
    transform :

    Methods
    -------
    __len__():
        Returns the length of the training dataset

    __getitem__():
        Returns the next (img, y_label) tuple, with the img as a transformed
        tensor
    '''
    def __init__(self, annotations_path, root_dir):
        self.annotations = pd.read_csv(annotations_path)
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Grayscale(), # 3-band to 1-band
            transforms.Resize((15, 12)), # average size of each char image
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        img = Image.open(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))
        img = self.transform(img)

        return (img, y_label)
