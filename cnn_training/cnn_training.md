# Embedding images with CNN

```python
%load_ext autoreload
%autoreload 2

import torch, torchvision
from pathlib import Path
from itertools import islice
import numpy as np
import PIL

dir_root = Path().resolve().parent
import sys; sys.path.append(str(dir_root))
from settings import Dir, Params
import utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device
```

## Data observation

```python
utils.plots(Dir.images.rglob("*.jpg"),
            lambda _, f: (Dir.captions / f.relative_to(Dir.images)
                         .with_suffix(".txt")).read_text()[:40])
```

## Load dataset

```python
import torchvision.transforms as transforms

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, mode): # mode is "train", "validate" or "test"
        self.dir_target = Dir.caption_vectors / mode
        self.data = list(self.dir_target.rglob("*.txt"))
        self.preprocess = transforms.Compose([
#             transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
        ])

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        file_caption = self.data[i]
        file_image = Dir.images / file_caption.relative_to(self.dir_target).with_suffix(".jpg")
        image = PIL.Image.open(file_image).convert('RGB')
        caption_vector = torch.from_numpy(np.loadtxt(str(file_caption)))
        return self.preprocess(image).float(), caption_vector.float(), str(file_image), str(file_caption)
```

```python
dataloaders = dict()
for mode in "validate", "train", "test":
    dataloaders[mode] = torch.utils.data.DataLoader(ImageDataset(mode), batch_size=12, num_workers=Params.workers, shuffle=True, pin_memory=True)
    print(f"{mode}: {len(dataloaders[mode])}x{dataloaders[mode].batch_size} = {len(dataloaders[mode])*dataloaders[mode].batch_size}")
```

## Use ResNet pretrained on ImageNet


Let's see some predictions of the ResNet on our images:

```python
import urllib
urllib.request.urlretrieve("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")

resnet = torchvision.models.resnet34(pretrained=True)
images, _, file_images, _ = next(iter(dataloaders["validate"]))
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
predictions = resnet(images)
utils.plots(file_images,
            lambda i, _: categories[torch.topk(predictions[i], 1)[1][0]])
```

# Model Fine-tuning

```python
from torch.utils.tensorboard import SummaryWriter

def train_model(model, device, criterion, optimizer, dataloaders, num_epochs=5):
    writer = SummaryWriter()
    for epoch in range(1, num_epochs+1):
        print(f"Epoch {epoch}/{num_epochs}")
        for phase in ['validate', 'train']:
            running_loss = .0
            if phase == 'train': model.train()
            else: model.eval()
            for inputs, labels, _, _ in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item()
            writer.add_scalar("Loss/" + "train" if phase == 'train' else "validate",
                              running_loss / len(dataloaders[phase]), epoch)
            print(f"{phase} Loss: {running_loss / len(dataloaders[phase]):.4f}")
    writer.flush()
```

```python
resnet = torchvision.models.resnet18(pretrained=True)
for param in resnet.parameters():
    param.requires_grad = False

resnet.fc = torch.nn.Linear(resnet.fc.in_features, Params.dim_embedding) # the model should output in the word vector space
resnet = resnet.to(device)
```

```python
# criterion = torch.nn.BCEWithLogitsLoss().to(device)
criterion = torch.nn.MSELoss(reduction="sum").to(device)
# optimizer = torch.optim.SGD(resnet.fc.parameters(), .01, momentum=.9)
optimizer = torch.optim.Adam(resnet.fc.parameters(), lr=0.01)
# torch.optim.SGD(resnet.parameters(), .01, momentum=.9, weight_decay=1e-4)
train_model(resnet, device, criterion, optimizer, dataloaders, num_epochs=3)
```

```python
# for param in resnet.parameters():
#     param.requires_grad = True
for m in list(resnet.modules())[-5:]:
    m.requires_grad_(True)

optimizer = torch.optim.Adam(resnet.parameters(), lr=1e-5)
#exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
train_model(resnet, device, criterion, optimizer, dataloaders, num_epochs=2)
```

```python
optimizer = torch.optim.Adam(resnet.parameters(), lr=1e-6)
#exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
train_model(resnet, device, criterion, optimizer, dataloaders, num_epochs=1)
```

```python
torch.save({'model_state_dict': resnet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, 
            f"models/resnet18__{Params.dim_embedding}_{Params.samples}.pt")
```

```python
utils.rm_dir(Dir.image_vectors)

with torch.no_grad():
    resnet.eval()
    for img, _, img_name, _ in dataloaders["test"]:
        vectors = resnet(img.to("cuda"))
        for i, v in enumerate(vectors):
            path = Dir.image_vectors / Path(img_name[i]).relative_to(Dir.images)
            path.parent.mkdir(exist_ok=True, parents=True)
            torch.save(v, str(path))
```

```python
import heapq as hq
from gensim.models import Word2Vec

word2vec = Word2Vec.load(str(Dir.model_embedding))
query_vector = torch.from_numpy(word2vec.wv.get_vector("wedding")).to("cuda")
closest = []
n_results = 5

for file_img in Dir.image_vectors.rglob("*.jpg"):
    v = torch.load(file_img)
    d = ((v - query_vector)**2).sum(axis=0).item()
    if len(closest) < n_results:
        hq.heappush(closest, (-d, file_img))
    elif -closest[0][0] > d:
        hq.heappushpop(closest, (-d, file_img))

dist, images = zip(*closest)
utils.plots([Dir.images / img.relative_to(Dir.image_vectors) for img in images],
            lambda i, _: -dist[i])
```

```python
torch.save(resnet.state_dict(), "resnet")
```

```python

```