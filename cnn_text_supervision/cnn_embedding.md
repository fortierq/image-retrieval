# Embedding images with CNN

```python
%load_ext autoreload
%autoreload 2

import matplotlib.pyplot as plt
import PIL
import torch, torchvision
from pathlib import Path
import numpy as np

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

## Load dataset

```python
import torchvision.transforms as transforms
import PIL

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, mode): # mode is "train", "validate" or "test"
        self.dir_images = dir_images
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

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        f = self.data[i]
        img = PIL.Image.open(self.dir_images / f.with_suffix(".jpg")).convert('RGB')
        caption_vector = torch.from_numpy(np.loadtxt(str(self.dir_target / f)))
        return self.preprocess(img), caption_vector
```

```python
dataset = ImageDataset("train")
print(f"Shape of an image: {dataset[0][0].shape}")
print(f"Shape of an embedded caption vector: {dataset[0][1].shape}")
ndim = dataset[0][1].shape[0]
```

## Use ResNet50 pretrained on ImageNet

```python
resnet = torchvision.models.resnet50(pretrained=True)
```

Let's see some predictions of the ResNet on our images:

```python
!wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt

img, vector = next(iter(dataloaders["validate"]))
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(resnet(img)[0], 5)

figure = plt.figure(figsize=(16, 4))
predictions = resnet(img)
for i, p in islice(enumerate(predictions), 4):
    figure.add_subplot(1, 4, i+1)
    plt.axis("off")
    top5_prob, top5_catid = torch.topk(p, 1)
    plt.title(categories[top5_catid[0]])
    plt.imshow(img[i].permute(1, 2, 0).clip(.0, 1.))
```

# Model Fine-tuning

```python

```

```python
def train_model(model, device, criterion, optimizer, dataloaders, num_epochs=5):
    for epoch in range(1, num_epochs):
        print(f"\nEpoch {epoch}/{num_epochs}")
        for phase in ['train', 'validate']:
            running_loss = .0
            if phase == 'train': model.train()
            else: model.eval()
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item()
            print(f"{phase} Loss: {running_loss / len(dataloaders[phase]):.4f}")
    return model
```

```python
resnet = torchvision.models.resnet18(pretrained=True)
resnet.fc = torch.nn.Linear(resnet.fc.in_features, ndim) # the model should output in the word vector space
resnet = resnet.to(device)
if torch.cuda.device_count() > 1:
    resnet = torch.nn.DataParallel(resnet)

dataloaders = dict()
for mode in "validate", "train":
    dataloaders[mode] = torch.utils.data.DataLoader(ImageDataset(mode), batch_size=8, num_workers=8, shuffle=True, pin_memory=True)
    print(f"{mode}: {len(dataloaders[mode])} batchs of size {dataloaders[mode].batch_size}")
criterion = torch.nn.BCEWithLogitsLoss().to(device)
optimizer = torch.optim.SGD(resnet.parameters(), .01, momentum=.9, weight_decay=1e-4) #torch.optim.Adam(resnet.parameters(), lr=0.01)
model = train_model(resnet, device, criterion, optimizer, dataloaders, num_epochs=20)
```

```python

```
