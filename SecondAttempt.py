
# coding: utf-8

# In[21]:


import torchvision
import torch
from torch.utils.data import DataLoader
from torchvision.datasets.folder import ImageFolder
from torchvision import transforms
from torchvision import models
import torch.nn as nn
from torch import optim
import time

batch_size = 32
nb_categories = 101


# In[22]:


data_root = '/home/wenjian/data/'

train_set_dir = data_root + 'wiki_crop_train/'
val_set_dir = data_root + 'wiki_crop_val/'
test_set_dir = data_root + 'wiki_crop_test/'

result_dir = '/home/wenjian/results/'


# In[23]:


train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])]) # Imagenet standards

val_transform = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])]) # Imagenet standards

# What should we do for test??????
test_transform = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor()])

train_set = ImageFolder(train_set_dir, transform=train_transform, target_transform=None)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

val_set = ImageFolder(val_set_dir, transform=val_transform, target_transform=None)
val_loader = DataLoader(val_set, batch_size=batch_size)

test_set = ImageFolder(test_set_dir, transform=test_transform, target_transform=None)
test_loader = DataLoader(test_set, batch_size=batch_size)


# In[24]:


model = models.vgg16(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
# The original classifier[6]'s input features
n_inputs = model.classifier[3].out_features
# Add on classifier
model.classifier[6] = nn.Sequential(
                      nn.Linear(n_inputs, 512), 
                      nn.ReLU(), 
                      nn.Dropout(0.4),
                      nn.Linear(512, nb_categories),                   
                      nn.Softmax())
  


# In[25]:


criterion = nn.KLDivLoss()
optimizer = optim.Adam(model.parameters())


# In[26]:


def linear_shadow(label_batch, gap=0.2, nb_categories=101):
    batch_size = label_batch.shape[0]
    l = torch.zeros(batch_size, nb_categories, dtype= torch.float)
    for i in range(batch_size):
        label = torch.tensor(label_batch[i], dtype=torch.float)
        for j in range(nb_categories):
            l[i,j] = 1- abs(label-j)*gap
            if l[i,j]<0:
                l[i,j]=0
    return l


# In[27]:


def target_distribution(vectors):
    # To normalize the distribution. Shape of the vectors are expected as (batch_size, vector_dim)
    # Noted that the softmax function in Pytorch only work for float type tensor
    return nn.functional.softmax(vectors, dim=1)


# https://github.com/pytorch/examples/blob/master/mnist/main.py

# In[28]:


def train(model, train_loader, criterion, optimizer, epoch):
    print('Start training...')
    model.train()
    log_interval = 10
    losses = []
    with open(result_dir + 'training_log.txt', 'a') as f: 
        for batch_idx, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            out = model(data)
            log_proba = torch.log(out)
            labels = linear_shadow(labels)
            labels = target_distribution(labels) # Normalize the distribution
            loss = criterion(log_proba, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                f.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            losses.append(loss.item())
    return losses


# In[29]:


def validate(model, val_loader):
    print("Start validation...")
    model.eval()
    test_loss = 0.0
    diff = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            out = model(data)
            log_proba = torch.log(out)
            labels = linear_shadow(target)
            labels = target_distribution(labels) # Normalize th distribution
            test_loss += criterion(log_proba, labels)
            preds = out.argmax(dim=1, keepdim=True)
            diff += nn.functional.l1_loss(preds, target)
    test_loss /= len(test_loader.dataset)
    diff /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, diff, len(test_loader.dataset)))
    with open(result_dir + 'validation_log.txt', 'a') as f:
        f.write('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, diff, len(test_loader.dataset)))
    return test_loss, diff


# In[30]:


n_epochs = 20
train_losses_all = []
val_loss_all = []
val_diff_all = []
for epoch in range(n_epochs):
    train_losses = train(model, train_loader, criterion, optimizer, epoch)
    train_losses_all += train_losses
    val_loss, diff = validate(model, test_loader)
    val_loss_all += val_loss
    val_diff_all += diff
torch.save(model.state_dict(),"model_trained.pt")

import numpy as np
train_loss_array = np.array(train_losses_all)
val_loss_array = np.array(val_loss_all)
val_diff_array = np.array(val_diff_all)

np.save(result_dir + 'train_loss_array.npy', train_loss_array)
np.save(result_dir + 'val_loss_array.npy', val_loss_array)
np.save(result_dir + 'val_diff_array.npy', val_diff_array)

