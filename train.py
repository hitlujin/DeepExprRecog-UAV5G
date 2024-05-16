import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models


DATASET_PATH = "ExpW_image_align_filtrate"
BATCH_SIZE = 64
EPOCHS = 30
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Dataset from subfolder
train_dataset = ImageFolder(DATASET_PATH, transform=train_transforms)
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# resnet50
model = models.resnet50(pretrained=True)
model.fc = torch.nn.Linear(2048, 7)
model = model.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


for epoch in range(EPOCHS):
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        # forward
        logits = model(data)
        loss = loss_fn(logits, target)
        # backward
        optimizer.zero_grad()
        loss.backward()
        # update
        optimizer.step()
        acc = (logits.argmax(dim=1) == target).float().mean()
        # print accuracy and loss every 100 batches
        if batch_idx % 100 == 0:
            print("Epoch: {} | Batch: {} | Loss: {} | Accuracy: {}".format(epoch, batch_idx, loss.item(), acc.item()))
        
torch.save(model.state_dict(), 'model.pt')

