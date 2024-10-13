
import os
import pandas as pd


import torch
from torchvision.io import read_image
from torch.utils.data import Dataset

#CODE FOR PREPARING DATA FOR TRAINING (i.e. "attaching" images to their classes/labels)


#define dataset class
class ZTF_lightkurve_img(Dataset): #This is the equivalent of the Tensorflow generator for pytorch that Salman mentioned
    def __init__(self, annotations_file, img_dir, transform = None,
                 target_transform = None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, str(self.img_labels.iloc[idx, 1]) + '.jpg')

        # Check if the file exists
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            # Return a placeholder image and label
            image = torch.zeros((3, 400, 400), dtype=torch.float32)  # Adjust size as needed
            label = torch.zeros(len(self.img_labels.columns) - 2, dtype=torch.float32)  # Adjust size as needed
        else:
            image = read_image(img_path).float()
            label = torch.tensor(list(self.img_labels.iloc[idx, 2:]), dtype=torch.float32) #each label is a list instead of a number [20:73]

        if self.transform:
            image = self.transform(image)
        #if self.target_transform:
        #    label = self.target_transform(label)
        label = torch.tensor(label)  # Convert label to tensor
        return image, label
