# Embedding images with CNN

```python
import matplotlib.pyplot as plt
import PIL
import torch, torchvision
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

## Load dataset

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
dataset = ImageDataset("train")
print(f"Shape of an image: {dataset[0][0].shape}")
print(f"Shape of an embedded caption vector: {dataset[0][1].shape}")
ndim = dataset[0][1].shape[0]
```

```python
dataloaders = dict()
for mode in "validate", "train", "test":
    dataloaders[mode] = torch.utils.data.DataLoader(ImageDataset(mode), batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
```

## Use ResNet50 pretrained on ImageNet

```python
resnet = torchvision.models.resnet50(pretrained=True)
```

Let's see some predictions of the ResNet on our images:

```python
!wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt

img, vector = next(iter(dataloaders["test"]))
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
resnet.fc = torch.nn.Linear(resnet.fc.in_features, ndim) # the model should output in the word vector space
resnet = resnet.to(device)
```

```python
import copy

def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=5):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        for phase in ['train', 'validate']:
            if phase == 'train': model.train()
            else: model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train': scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            if phase == 'validate' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()
    print(f"Best val Acc: {best_acc:4f}")
    model.load_state_dict(best_model_wts)
    return model
```

```python
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
model = train_model(resnet, criterion, optimizer, exp_lr_scheduler, dataloaders)
```

```python

```
