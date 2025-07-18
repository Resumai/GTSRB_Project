import os
from PIL import Image
from torch.utils.data import Dataset
from utils.image_transforms import get_default_transform


class GTSRBImageLoader(Dataset):
    def __init__(self, root_dir, transform=None, unlabeled=False):
        self.transform = transform
        self.root_dir = root_dir
        self.samples = []
        self.unlabeled = unlabeled

        # Traverse folders like 00000/, 00001/...
        for item_name in sorted(os.listdir(root_dir)):
            # For unlabeled data loading, for Final_Test
            if self.unlabeled and item_name.endswith(".ppm"):
                item_path = os.path.join(root_dir, item_name)
                self.samples.append((item_path, item_name))
                continue
            
            # Define item 
            item_path = os.path.join(root_dir, item_name)

            # Makes sure it's a folder, if not - skip current itteration
            if not os.path.isdir(item_path):
                continue

            # Traverse thru given item_path folders
            for file_name in os.listdir(item_path):
                if file_name.endswith(".ppm"):
                    full_file_path = os.path.join(item_path, file_name)
                    self.samples.append((full_file_path, int(item_name)))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        full_file_path, label_or_fname = self.samples[index]
        image = Image.open(full_file_path).convert("RGB") # Convert to Rapid Grenade Launcher :)

        if self.transform:
            image = self.transform(image)

        return image, label_or_fname # Can be class ID or filename depending on unlabeled=True