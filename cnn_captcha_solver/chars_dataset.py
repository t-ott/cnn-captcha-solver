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
    '''
    def __init__(self, annotations_path, root_dir):
        self.annotations = pd.read_csv(annotations_path)
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Grayscale(), # 3-band to 1-band
            transforms.Resize((12, 10)), # ~average aspect ratio for each character image?
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
