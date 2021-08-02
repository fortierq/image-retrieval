# Embedding images with CNN

```python
%load_ext autoreload
%autoreload 2

import matplotlib.pyplot as plt
import PIL
import torch, torchvision
from pathlib import Path
from itertools import islice
import numpy as np

dir_root = Path().resolve().parent
dir_word_embedding = dir_root / "word2vec_model"
dir_caption_vectors = dir_word_embedding / "vectors"
dir_images = dir_root / "data" / "img_resized"
dir_captions = dir_root / "data" / "captions"
dir_image_vectors = dir_root / "cnn_text_supervision" / "vectors"

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
        self.dir_target = dir_caption_vectors / mode
        self.data = [f.relative_to(self.dir_target) 
                     for f in self.dir_target.rglob("*.txt")]
        self.preprocess = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
        ])

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        f = self.data[i]
        img = PIL.Image.open(dir_images / f.with_suffix(".jpg")).convert('RGB')
        caption_vector = torch.from_numpy(np.loadtxt(str(self.dir_target / f)))
        return str(f), self.preprocess(img).float(), caption_vector.float()
```

```python
dataset = ImageDataset("train")
print(f"Shape of an image: {dataset[0][1].shape}")
print(f"Shape of an embedded caption vector: {dataset[0][2].shape}")
ndim = dataset[0][1].shape[0]
```

## Use ResNet pretrained on ImageNet


Let's see some predictions of the ResNet on our images:

```python
!wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt

resnet = torchvision.models.resnet50(pretrained=True)
dataloaders = dict()
for mode in "validate", "train":
    dataloaders[mode] = torch.utils.data.DataLoader(ImageDataset(mode), batch_size=12, num_workers=8, shuffle=True, pin_memory=True)
    print(f"{mode}: {len(dataloaders[mode])} batchs of size {dataloaders[mode].batch_size}")

_, img, vector = next(iter(dataloaders["validate"]))
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(resnet(img)[0], 5)

figure = plt.figure(figsize=(20, 4))
predictions = resnet(img)
for i, p in islice(enumerate(predictions), 5):
    print(p[0])
    figure.add_subplot(1, 5, i+1)
    plt.axis("off")
    top5_prob, top5_catid = torch.topk(p, 1)
    plt.title(categories[top5_catid[0]])
    plt.imshow(img[i].permute(1, 2, 0).clip(.0, 1.))
```

# Model Fine-tuning

```python
def train_model(model, device, criterion, optimizer, dataloaders, num_epochs=5):
    for epoch in range(1, num_epochs+1):
        print(f"Epoch {epoch}/{num_epochs}")
        for phase in ['validate', 'train']:
            running_loss = .0
            if phase == 'train': model.train()
            else: model.eval()
            for _, inputs, labels in dataloaders[phase]:
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
```

```python
resnet = torchvision.models.resnet18(pretrained=True)
for param in resnet.parameters():
    param.requires_grad = False

resnet.fc = torch.nn.Linear(resnet.fc.in_features, 40) # the model should output in the word vector space
resnet = resnet.to(device)
    
dataloaders = dict()
for mode in "validate", "train", "test":
    dataloaders[mode] = torch.utils.data.DataLoader(ImageDataset(mode), batch_size=10, num_workers=8, shuffle=True, pin_memory=True)
    print(f"{mode}: {len(dataloaders[mode])} batchs of size {dataloaders[mode].batch_size}")
```

```python
# criterion = torch.nn.BCEWithLogitsLoss().to(device)
criterion = torch.nn.MSELoss(reduction="sum").to(device)
# optimizer = torch.optim.SGD(resnet.fc.parameters(), .01, momentum=.9)
optimizer = torch.optim.Adam(resnet.fc.parameters(), lr=0.01)
# torch.optim.SGD(resnet.parameters(), .01, momentum=.9, weight_decay=1e-4)
train_model(resnet, device, criterion, optimizer, dataloaders, num_epochs=10)
```

```python
# for param in resnet.parameters():
#     param.requires_grad = True
for m in list(resnet.modules())[-5:]:
    m.requires_grad_(True)

optimizer = torch.optim.Adam(resnet.parameters(), lr=1e-5)
#exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
train_model(resnet, device, criterion, optimizer, dataloaders, num_epochs=10)
```

```python
optimizer = torch.optim.Adam(resnet.parameters(), lr=1e-6)
#exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
train_model(resnet, device, criterion, optimizer, dataloaders, num_epochs=30)
```

```python
for img_name, img, _ in dataloaders["test"]:
    vectors = resnet(img.to("cuda")))
    for i, v in enumerate(vectors):
        path = dir_image_vectors / img_name[i]
        path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(v, str(path))
```

```python
import heapq as hq
from gensim.models import Word2Vec

word2vec = Word2Vec.load(str(dir_word_embedding / "model_captions"))
query_vector = torch.from_numpy(word2vec.wv.get_vector("cake")).to("cuda")
closest = []
n_results = 5

for file_img in dir_image_vectors.rglob("*.txt"):
    v = torch.load(file_img)
    d = ((v - query_vector)**2).sum(axis=0).item()
    if len(closest) < n_results:
        hq.heappush(closest, (-d, file_img))
    elif -closest[0][0] > d:
        hq.heappushpop(closest, (-d, file_img))

figure = plt.figure(figsize=(20, 4))
for i, (nd, file_img) in enumerate(closest):
    figure.add_subplot(1, 5, i+1)
    plt.axis("off")
    path = dir_images / file_img.relative_to(dir_image_vectors).with_suffix(".jpg")
    plt.imshow(PIL.Image.open(path).convert('RGB'))
    plt.title(str(-nd))
```

```python
torch.save(resnet.state_dict(), "resnet")
```
