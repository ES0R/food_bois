# data_loader.py
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class FoodDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['Image_Name'] != '#NAME?']  # Filter out invalid image names
        self.df = self.df[self.df['Image_Name'] != 'pan-seared-salmon-on-baby-arugula-242445']  # Filter out invalid image names
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.df.iloc[idx, 4] + '.jpg')  # Index 3 for 'Image_Name'
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        title = str(self.df.iloc[idx, 1])  # Index 0 for 'Title'
        ingredients = str(self.df.iloc[idx, 2])  # Index 1 for 'Ingredients'
        instructions = str(self.df.iloc[idx, 3])  # Index 2 for 'Instructions'
        combined_text = f"{title} {ingredients} {instructions}"

        return {'image': image, 'text': combined_text, 'image_name': self.df.iloc[idx, 4]}
