import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

# Function for loading notMNIST Dataset
def loadData(datafile = "notMNIST.npz"):
    with np.load(datafile) as data:
        Data, Target = data["images"].astype(np.float32), data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

# Custom Dataset class. 
class notMNIST(Dataset):
    def __init__(self, annotations, images, transform=None, target_transform=None):
        self.img_labels = annotations
        self.imgs = images
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.imgs[idx]
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# Define CNN
class CNN(nn.Module):
    def __init__(self, drop_out_p=0.0):
        super(CNN, self).__init__()

        # TODO
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4)
        self.bn1 = nn.BatchNorm2d(32)  # Batch Normalization 1
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))  # Max Pooling

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4)
        self.bn2 = nn.BatchNorm2d(64)  # Batch Normalization 2
        self.d1 = nn.Linear(1024, 784)

        self.drop = nn.Dropout(p=drop_out_p)
        self.d2 = nn.Linear(784, 10)

    def forward(self, x):
        # TODO
        # conv1 --> ReLU --> BatchNorm1 --> Max pooling
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        # conv2 --> ReLU --> BatchNorm2 --> Max pooling
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.maxpool(x)

        # flatten --> drop --> ReLU
        x = x.flatten(start_dim=1)
        x = self.drop(x)
        x = self.d1(x)
        x = F.relu(x)

        # Fully connected layer
        x = self.d2(x)

        return x


# Define FNN
class FNN(nn.Module):
    def __init__(self, drop_out_p=0.0):
        super(FNN, self).__init__()

        # TODO
        self.d1 = nn.Linear(784, 10)
        self.d2 = nn.Linear(10, 10)
        self.drop = nn.Dropout(p=drop_out_p)

    def forward(self, x):
        # TODO
        # flatten => 32 x 784
        x = x.flatten(start_dim=1)

        # 32 x 784 => 32 x 10
        x = self.d1(x)
        x = F.relu(x)

        # 32 x 10 => 32 x 10
        x = self.d2(x)
        x = F.relu(x)

        # drop
        x = self.drop(x)

        return x


# Commented out IPython magic to ensure Python compatibility.
# Compute accuracy
def get_accuracy(model, dataloader):
    model.eval()
    device = next(model.parameters()).device
    corrects = 0

    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # TODO Calculate Accuracy
            corrects += (torch.max(model(images), 1)[1].view(labels.size()).data == labels.data).sum()

    # Get Accuracy
    accuracy = 100.0 * (corrects / len(dataloader.dataset))
    return accuracy


def train(model, device, learning_rate, weight_decay, train_loader, val_loader, test_loader, num_epochs=50,
          verbose=False):
    # TODO
    # Define your cross entropy loss function here
    # Use cross entropy loss
    criterion = nn.CrossEntropyLoss()

    # TODO
    # Define your optimizer here
    # Use AdamW optimizer, set the weights, learning rate and weight decay argument.
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    acc_hist = {'train': [], 'val': [], 'test': []}

    for epoch in range(num_epochs):
        # train_running_loss = 0.0
        # train_acc = 0.0

        model = model.train()
        ## training step
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # TODO
            # Follow the step in the tutorial
            ## forward + backprop + loss
            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()

            ## update model params
            optimizer.step()

            # train_running_loss += loss.detach().item()
            # train_acc += get_accuracy(model, train_loader)

        model.eval()
        acc_hist['train'].append(get_accuracy(model, train_loader))
        acc_hist['val'].append(get_accuracy(model, val_loader))
        acc_hist['test'].append(get_accuracy(model, test_loader))

        if verbose:
            print('Epoch: %d | Train Accuracy: %.2f | Validation Accuracy: %.2f | Test Accuracy: %.2f' \
                  % (epoch, acc_hist['train'][-1], acc_hist['val'][-1], acc_hist['test'][-1]))

    return model, acc_hist


def experiment(model_type='CNN', learning_rate=0.0001, dropout_rate=0.0, weight_decay=0.0, num_epochs=50,
               verbose=False):
    # Use GPU if it is available.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Inpute Batch size:
    BATCH_SIZE = 32

    # Convert images to tensor
    transform = transforms.Compose(
        [transforms.ToTensor()])

    # Get train, validation and test data loader.
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

    train_data = notMNIST(trainTarget, trainData, transform=transform)
    val_data = notMNIST(validTarget, validData, transform=transform)
    test_data = notMNIST(testTarget, testData, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # Specify which model to use
    if model_type == 'CNN':
        model = CNN(dropout_rate)
    elif model_type == 'FNN':
        model = FNN(dropout_rate)

    # Loading model into device
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    model, acc_hist = train(model, device, learning_rate, weight_decay, train_loader, val_loader, test_loader,
                            num_epochs=num_epochs, verbose=verbose)

    # Release the model from the GPU (else the memory wont hold up)
    model.cpu()

    return model, acc_hist

def compare_arch():
    # Run the experiment, and obtain the accuracy output
    CNN_output = experiment(model_type='CNN', learning_rate=0.0001, dropout_rate=0.0, weight_decay=0.0, num_epochs=50, verbose=False)
    FNN_output = experiment(model_type='FNN', learning_rate=0.0001, dropout_rate=0.0, weight_decay=0.0, num_epochs=50, verbose=False)

    #Turn tensors into list
    x = list(range(0, 50))
    CNN_train_acc_list = []
    CNN_test_acc_list = []
    FNN_train_acc_list = []
    FNN_test_acc_list = []

    # CNN Model Result
    for e in CNN_output[1]['train']:
        CNN_train_acc_list.append(e.item())

    for e in CNN_output[1]['test']:
        CNN_test_acc_list.append(e.item())

    # FNN Model Result
    for e in FNN_output[1]['train']:
        FNN_train_acc_list.append(e.item())

    for e in FNN_output[1]['test']:
        FNN_test_acc_list.append(e.item())

    # Plot the trainning accuracy respect to epoches
    plt.plot(x,CNN_train_acc_list,'b-')
    plt.plot(x,CNN_test_acc_list,'r-')
    plt.plot(x,FNN_train_acc_list,'b--')
    plt.plot(x,FNN_test_acc_list,'r--')

    plt.legend(['CNN_train','CNN_test','FNN_train','FNN_test'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy in %')
    plt.title('Compare Arch Plot')


def compare_dropout():
    # Run the experiment, and obtain the accuracy output
    CNN_output_d05 = experiment(model_type='CNN', learning_rate=0.0001, dropout_rate=0.5, weight_decay=0.0,
                                num_epochs=50, verbose=False)  # with dropout rate of 0.5
    CNN_output_d08 = experiment(model_type='CNN', learning_rate=0.0001, dropout_rate=0.8, weight_decay=0.0,
                                num_epochs=50, verbose=False)  # with dropout rate of 0.8
    CNN_output_d095 = experiment(model_type='CNN', learning_rate=0.0001, dropout_rate=0.95, weight_decay=0.0,
                                 num_epochs=50, verbose=False)  # with dropout rate of 0.95

    # Turn tensors into list
    x = list(range(0, 50))
    CNN_d05_train_acc_list = []
    CNN_d05_test_acc_list = []
    CNN_d08_train_acc_list = []
    CNN_d08_test_acc_list = []
    CNN_d095_train_acc_list = []
    CNN_d095_test_acc_list = []

    # dropout rate of 0.5
    for e in CNN_output_d05[1]['train']:
        CNN_d05_train_acc_list.append(e.item())

    for e in CNN_output_d05[1]['test']:
        CNN_d05_test_acc_list.append(e.item())

    # dropout rate of 0.8
    for e in CNN_output_d08[1]['train']:
        CNN_d08_train_acc_list.append(e.item())

    for e in CNN_output_d08[1]['test']:
        CNN_d08_test_acc_list.append(e.item())

    # dropout rate of 0.95
    for e in CNN_output_d095[1]['train']:
        CNN_d095_train_acc_list.append(e.item())

    for e in CNN_output_d095[1]['test']:
        CNN_d095_test_acc_list.append(e.item())

    # Plot the trainning accuracy respect to epoches
    plt.plot(x, CNN_d05_train_acc_list, 'b-')
    plt.plot(x, CNN_d08_train_acc_list, 'r-')
    plt.plot(x, CNN_d095_train_acc_list, 'g-')
    plt.plot(x, CNN_d05_test_acc_list, 'b--')
    plt.plot(x, CNN_d08_test_acc_list, 'r--')
    plt.plot(x, CNN_d095_test_acc_list, 'g--')

    plt.legend(['CNN_train_dr=0.5', 'CNN_train_dr=0.8', 'CNN_train_dr=0.95', 'CNN_test_dr=0.5', 'CNN_test_dr=0.8',
                'CNN_test_dr=0.95'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy in %')
    plt.title('Compare Dropout Rate Plot')


def compare_l2():
    # Run the experiment, and obtain the accuracy output
    CNN_output_w01 = experiment(model_type='CNN', learning_rate=0.0001, dropout_rate=0.0, weight_decay=0.1,
                                num_epochs=50, verbose=False)  # with weight decay of 0.1
    CNN_output_w1 = experiment(model_type='CNN', learning_rate=0.0001, dropout_rate=0.0, weight_decay=1.0,
                               num_epochs=50, verbose=False)  # with weight decay of 1.0
    CNN_output_w10 = experiment(model_type='CNN', learning_rate=0.0001, dropout_rate=0.0, weight_decay=10.0,
                                num_epochs=50, verbose=False)  # with weight decay of 10.0

    # Turn tensors into list
    x = list(range(0, 50))
    CNN_w01_train_acc_list = []
    CNN_w01_test_acc_list = []
    CNN_w1_train_acc_list = []
    CNN_w1_test_acc_list = []
    CNN_w10_train_acc_list = []
    CNN_w10_test_acc_list = []

    # weight decay of 0.1
    for e in CNN_output_w01[1]['train']:
        CNN_w01_train_acc_list.append(e.item())

    for e in CNN_output_w01[1]['test']:
        CNN_w01_test_acc_list.append(e.item())

    # weight decay of 1.0
    for e in CNN_output_w1[1]['train']:
        CNN_w1_train_acc_list.append(e.item())

    for e in CNN_output_w1[1]['test']:
        CNN_w1_test_acc_list.append(e.item())

    # weight decay of 10.0
    for e in CNN_output_w10[1]['train']:
        CNN_w10_train_acc_list.append(e.item())

    for e in CNN_output_w10[1]['test']:
        CNN_w10_test_acc_list.append(e.item())

    # Plot the trainning accuracy respect to epoches
    plt.plot(x, CNN_w01_train_acc_list, 'b-')
    plt.plot(x, CNN_w1_train_acc_list, 'r-')
    plt.plot(x, CNN_w10_train_acc_list, 'g-')
    plt.plot(x, CNN_w01_test_acc_list, 'b--')
    plt.plot(x, CNN_w1_test_acc_list, 'r--')
    plt.plot(x, CNN_w10_test_acc_list, 'g--')

    plt.legend(['CNN_train_wd=0.1', 'CNN_train_wd=1.0', 'CNN_train_wd=10.0', 'CNN_test_wd=0.1', 'CNN_test_wd=1.0',
                'CNN_test_wd=10.0'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy in %')
    plt.title('Compare Weight Decay Plot')
