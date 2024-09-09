import numpy as np
from matplotlib import pyplot as plt

from CFA import CFA
from metric import MSE, MaskedMSELoss
from model.unet_model import UNet, model
from utils.dataset import DATA_Loader
from torch import optim
import torch.nn as nn
import torch

from utils.optimizer import create_lr_scheduler


class Criterion(nn.Module):
    def __init__(self, step=3):
        super(Criterion, self).__init__()
        self.step = step
        self.criterions = nn.ModuleList([nn.L1Loss(reduction='mean') for i in range(step)])

    def forward(self, input, target):
        loss_sum = self.criterions[0](input[-1], target)
        return loss_sum


def train_net(net, device, mask, train_loader, criterion,epochs=100, warmup=10, lr=0.003):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)

    scheduler = create_lr_scheduler(optimizer, len(train_loader), epochs, warmup_epochs=warmup)
    best_loss = float('inf')
    losses = []
    net.mask = net.mask.to(device=device, dtype=torch.float32)


    for epoch in range(epochs):
        mean_loss = 0
        net.train()
        for image, label in train_loader:
            optimizer.zero_grad()
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            hidden = net(image, mask)
            loss = criterion(hidden, label)
            mean_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
        print(f'epoch {epoch + 1} Loss/train', mean_loss / len(train_loader), " lr: ", optimizer.param_groups[0]["lr"])
        losses.append(mean_loss / len(train_loader))
        if mean_loss / len(train_loader) < best_loss:
            best_loss = mean_loss / len(train_loader)
            torch.save(net.state_dict(), 'best_layer4_dim24.pth')


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_path = r"E:\DATASET\McMaster"
    input_shape = [256, 256, 3]
    CFA_pattern = "bayer_rggb"
    epochs = 30
    warmup = 1
    batch_size = 1
    lr = 0.0003

    hidden_dim = 24
    layers = 6
    bilinear = False

    cfa = CFA(input_shape)
    mask = cfa.choose(CFA_pattern)
    net = model(n_channels=3, n_classes=3, mask=mask, dim=hidden_dim, layers=layers, bilinear=bilinear)
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total number of parameters: {total_params / 1e6} M")
    net.to(device=device)
    isbi_dataset = DATA_Loader(data_path, mask=mask, size=input_shape)
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    criterion = Criterion(layers)
    train_net(net, device, mask, train_loader,criterion, epochs=epochs, warmup=warmup, lr=lr)
