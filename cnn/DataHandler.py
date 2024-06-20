from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from torchvision import transforms

class DataHandler(Dataset):
    def __init__(self, datapath, txtpath, label_map=None):
        self.datapath = datapath
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.imgs = []
        with open(txtpath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    words = line.split()
                    self.imgs.append((words[0], words[1]))

        if label_map is None:
            self.label_map = {label: idx for idx, label in enumerate(set(label for _, label in self.imgs))}
            self.inv_label_map = {v: k for k, v in self.label_map.items()}
        else:
            self.label_map = label_map
            self.inv_label_map = {v: k for k, v in label_map.items()}

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        pic, label = self.imgs[index]
        pic = Image.open(os.path.join(self.datapath, pic)).convert('L')
        if self.transform:
            pic = self.transform(pic)
        label = self.label_map[label]
        return pic, label

    @staticmethod
    def load_data(train_datapath, train_txtpath, test_datapath, test_txtpath, batch_size=32):
        train_set = DataHandler(train_datapath, train_txtpath)
        test_set = DataHandler(test_datapath, test_txtpath, label_map=train_set.label_map)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)

        return train_loader, test_loader, train_set.inv_label_map
