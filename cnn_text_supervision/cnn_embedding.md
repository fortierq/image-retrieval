# Embedding images with CNN

```python
import matplotlib.pyplot as plt
import PIL
import torch
import torchvision
from pathlib import Path

dir_root = Path().resolve().parent
dir_caption_vectors = dir_root / "word2vec_model" / "vectors"
dir_images = dir_root / "data" / "img_resized"
dir_captions = dir_root / "data" / "captions"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device
```

## Data observation

```python
from itertools import islice

figure = plt.figure(figsize=(12, 12))
for i, f in enumerate(islice(dir_images.rglob("*.jpg"), 9)):
    figure.add_subplot(3, 3, i+1)
    plt.axis("off")
    caption = (dir_captions / f.relative_to(dir_images).with_suffix(".txt")).read_text()
    plt.title(caption[:30] + "\n" + caption[30:60])
    plt.imshow(PIL.Image.open(f))
```

## Define PyTorch Dataset

```python
from torchvision import transforms

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, mode): # mode is "train", "validate" or "test" 
        self.dir_target = dir_caption_vectors / mode
        self.data = [f.relative_to(self.dir_target) 
                     for f in self.dir_target.rglob("*.txt")]
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
        img = PIL.Image.open(dir_images / f.with_suffix(".jpg")).convert('RGB')
        caption_vector = torch.load(str(self.dir_target / f))
        return self.preprocess(img), caption_vector
```

```python
print(f"Shape of an image: {dataset[0][0].shape}")
print(f"Shape of an embedded caption vector: {dataset[0][1].shape}")
```

```python

```
