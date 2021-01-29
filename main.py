import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F

from model import Net
from dataloader import MnistDataset

def train_one_epoch(net, device, train_loader, optimizer, epoch, criterion, running_loss):
    net.train()
    for i, json_data in enumerate(train_loader):
        # print(data['image'].size(), data['label'])
        data, target = json_data['image'].to(device, dtype=torch.float), json_data['label'].to(device)

        optimizer.zero_grad()
        outputs = net(data) # 前向传递
        # loss = criterion(outputs, target)
        loss = F.nll_loss(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % print_freq == print_freq-1:
            print(i+1, running_loss/print_freq)
            running_loss = 0

def test(net, device, test_loader):
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for json_data in test_loader:
            data, target = json_data['image'].to(device, dtype=torch.float), json_data['label'].to(device)
            output = net(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    
if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    device = torch.device('cuda')

    train_csv_file = './NumberDetecor/datasets/mnist_train.csv'
    test_csv_file = './NumberDetecor/datasets/mnist_test.csv'
    root = './NumberDetecor/datasets'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    train_dataset = MnistDataset(train_csv_file, root)
    test_dataset = MnistDataset(test_csv_file, root)

    """
    train_dataset = datasets.MNIST(root, train=True, download=True, \
        transform = transform)
    test_dataset = datasets.MNIST(root, train=False, download=True, \
        transform = transform)
    """
    train_loader = DataLoader(train_dataset, batch_size=4, \
        shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, \
        shuffle=True)


    net = Net().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    running_loss = 0
    print_freq = 1000

    epoches = 2

    if os.path.exists(root+'./mnist_cnn.pt'):
        net = Net()
        net.load_state_dict(torch.load(root+'./mnist_cnn.pt'))
        test(net, device, test_loader)
    
    else:
        for epoch in range(epoches):
            train_one_epoch(net, device, train_loader, optimizer, epoch, criterion, running_loss)

            test(net, device, test_loader)
        torch.save(net.state_dict(), root+'./mnist_cnn.pt')