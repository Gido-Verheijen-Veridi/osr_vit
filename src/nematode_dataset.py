import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import numpy as np


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, val_transform=None, load_data = True):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = None
        self.targets = None
        self.in_out_targets = None
        self.val_transform = val_transform

        if load_data:
            self.image_paths, self.targets, self.in_out_targets = self._load_images()

    def _load_images(self):
        #in = 1, out = 0
        image_paths = []
        image_labels = []
        in_out_targets = []
        for folder_name in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder_name)
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    if file_name.endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append(os.path.join(folder_path, file_name))
                        if "Meloidogyne" in folder_name:
                            image_labels.append(0)
                            in_out_targets.append(1)
                        elif "Penetrans" in folder_name:
                            image_labels.append(1)
                            in_out_targets.append(1)
                        else:
                            image_labels.append(2) 
                            in_out_targets.append(0)
        
        return np.array(image_paths), torch.tensor(image_labels), torch.tensor(in_out_targets)

    def __len__(self):
        return len(self.image_paths)

    def get_item_path(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.image_paths[idx]
        image, img_label, in_out_target = self.__getitem__(idx)
        return image, img_label, img_path

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        img_label = self.targets[idx]
        in_out_target = self.in_out_targets[idx]
        image = Image.open(img_path).convert('RGB')  # Ensure 3-channel image
        if self.transform:
            image = self.transform(image)
        return image, img_label

class ImageDataset_Filter(ImageDataset):

    def __Filter__(self, in_set : bool):
        targets = self.targets.data.numpy()
        in_out_targets = self.in_out_targets.data.numpy()
        mask, new_targets, new_in_out_target = [], [], []
        for i in range(len(targets)):
            if self.in_out_targets[i] == int(in_set):
                mask.append(i)
                new_targets.append(targets[i])
                new_in_out_target.append(in_out_targets[i])
        self.targets = np.array(new_targets)
        mask = torch.tensor(mask).long().numpy()
        self.image_paths = self.image_paths[mask]#torch.index_select(self.image_idx, 0, mask)
        self.in_out_targets = np.array(new_in_out_target)

    def __Split__(self, ratios):
        idxs = torch.utils.data.random_split(torch.arange(0, len(self.targets)), ratios)
        new_datasets = []
        for n, idx in enumerate(idxs):
            if n != 0 and self.val_transform != None:
                ds = ImageDataset_Filter(self.root_dir, self.val_transform, load_data=False)
            else: 
                ds = ImageDataset_Filter(self.root_dir, self.transform, load_data=False)
            ds.image_paths = self.image_paths[idx]
            ds.targets = self.targets[idx]
            ds.in_out_targets = self.in_out_targets[idx]
            new_datasets.append(ds)
        return new_datasets

def merge_dataset(in_dataset : ImageDataset, out_dataset : ImageDataset):
    ds = ImageDataset(in_dataset.root_dir, out_dataset.transform, load_data=False)
    ds.image_paths = np.concatenate([in_dataset.image_paths, out_dataset.image_paths])
    ds.targets = np.concatenate([in_dataset.targets, out_dataset.targets])
    ds.in_out_targets = np.concatenate([in_dataset.in_out_targets, out_dataset.in_out_targets])
    return ds




def make_loaders(root_dir, batch_size, img_size, seed):

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # Resize images to 256x256
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize based on ImageNet
    ]) 

    dataset = ImageDataset_Filter(root_dir, transform=transform, val_transform=transform)
    dataset.__Filter__(True)
    torch.manual_seed(seed)
    train_set, val_set = dataset.__Split__([0.8, 0.2])
    
    #train_set.__Filter__(known=self.known)
    #print('All Train Data:', len(train_set))

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=0,
    )

    #val_set.__Filter__(known=self.known)
    #print('All Validation Data:', len(val_set))
        
    in_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=True,
        num_workers=0
    )

    out_set = ImageDataset_Filter(root_dir, transform=transform)
    out_set.__Filter__(False)
    
    #val_out = merge_dataset(val_set, out_set)

    out_loader = torch.utils.data.DataLoader(
        out_set, batch_size=batch_size, shuffle=True,
        num_workers=0
    )

    print('Train: ', len(train_set), 'Test: ', len(val_set), 'Out: ', len(out_set))
    #print('All Test: ', (len(val_out)))

    return train_loader, in_loader, out_loader

if __name__=="__main__":
    # Define any transforms (e.g., resizing, normalizing)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images to 256x256
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize based on ImageNet
    ])

    # Create Dataset and DataLoader
    train, val, out = make_loaders(r'D:\Veridi\Images', 4, transform, transform, 2502)
    # Example: Iterate through the DataLoader
    for images in val:
        print(images[0].shape, images[1].shape)
        print(images[1])  # Output the shape of the batch
        print(images[2])
        break