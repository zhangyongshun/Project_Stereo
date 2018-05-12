# coding=utf-8
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import numpy as np
import torch.optim as optim
import time
from tqdm import tqdm
from multiScaleSingleImageDepth.data_io import *
from multiScaleSingleImageDepth.model import *

def train_coarse(model, criterion, optimizer,dataset_loader,n_epochs,print_every):
    start = time.time()
    losses = []
    print("Training for %d epochs..." % n_epochs)
    for epoch in tqdm(range(1, n_epochs + 1)):
        loss = 0

        for data in dataset_loader:
            # get the inputs
            inputs=data["image"]
            depths=data["depth"]
            optimizer.zero_grad()
            # forward
            outputs = model(inputs).cuda()
            depths = depths.view(depths.size(0), -1)
            inputs, depths = Variable(inputs), Variable(depths)
            loss = criterion(outputs, depths)
            loss += loss.data[0]
            loss.backward()
            optimizer.step()
    return losses

def main():
    dataset = data(root_dir='./', transform=transforms.Compose([
                                                   Rescale((input_height, input_width),(output_height, output_width)),
                                                   ToTensor()]))
    dataset_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    coarse_net = CoarseNet().cuda()
    optimizer_coarse = optim.SGD(coarse_net.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    fine_net = FineNet()
    optimizer_fine = optim.SGD(fine_net.parameters(), lr=0.01)
    loss = train_coarse(coarse_net,criterion,optimizer_coarse,dataset_loader,5,1)

if __name__ == "__main__":
    main()
