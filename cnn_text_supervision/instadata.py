from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class InstaData(Dataset):
    def __init__(self, caption_type):
        self.data = []
        self.path_caption = Path("../word2vec_model/caption")
        self.caption_type = caption_type
        for f in (self.path_caption / self.caption_type).rglob("*.txt"):
            self.data.append(f.relative_to(self.path_caption / self.caption_type))
        self.preprocess = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.Resize(256), # If we want the crop to be more centered
            # transforms.CenterCrop(224), # This makes more sense for testing
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        f = self.data[i]
        img = Image.open(Path(f"../InstaCities1M/img_resized_1M/cities_instagram") / f.with_suffix(".jpg")).convert('RGB')
        caption = torch.from_numpy(np.loadtxt(str(self.path_caption / self.caption_type / f)))
        return f.stem, self.preprocess(img), caption
