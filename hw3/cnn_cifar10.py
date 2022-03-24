import torch, torchvision, numpy, pandas
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.datasets as data


train_data = data.CIFAR10('data/cifar10', train = True, transform = torchvision.transforms.ToTensor(), download = True)
test_data = data.CIFAR10('data/cifar10', train = False, transform = torchvision.transforms.ToTensor(), download = True)

train_loader = DataLoader(train_data, batch_size = 8, shuffle = True)
test_loader = DataLoader(test_data, batch_size = 8, shuffle = True)

# plt.imshow(train_data.data[0].numpy(), cmap = 'gray')
# plt.show()

class CNN(nn.Module):
    def __init__(self, input, output):
        super(CNN, self).__init__()
        self.conv_layer1 = nn.Conv2d(input, 16, 3, 1)
        self.relu_layer1 = nn.ReLU()
        self.maxpool_layer1 = nn.MaxPool2d(2)
        self.conv_layer2 = nn.Conv2d(16, 32, 3, 1)
        self.relu_layer2 = nn.ReLU()
        self.maxpool_layer2 = nn.MaxPool2d(2)
        self.function1 = nn.Linear(32*6*6, 100)
        self.function2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.relu_layer1(x)
        x = self.maxpool_layer1(x)
        x = self.conv_layer2(x)
        x = self.relu_layer2(x)
        x = self.maxpool_layer2(x)
        x = x.view(-1, 32*6*6)
        x = self.function1(x)
        x = self.function2(x)
        return x

device = torch.device('cpu')
model = CNN(3, 10)
model = model.to(device)
lossfunction =nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr = 1e-5)

def train():
    model.train()
    train_loss = 0
    train_acc = 0
    for index, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        pred = model(data)
        loss = lossfunction(pred, target)
        opt.zero_grad() 
        loss.backward()
        opt.step()
        train_loss += loss
        _, y = pred.max(1)
        correct = (y == target).sum().item()
        train_acc += correct / data.shape[0]
    
    return train_loss/len(train_loader), train_acc/len(train_loader)

def test():
    test_loss = 0
    test_acc = 0
    model.eval()
    for index, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        pred = model(data)
        loss = lossfunction(pred, target) 
        test_loss += loss
        _, y = pred.max(1)
        correct = (y == target).sum().item()
        test_acc += correct / data.shape[0]
    
    return test_loss/len(test_loader), test_acc/len(test_loader)

train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []
epoch = 2
for i in range(epoch):
    train_loss, train_acc = train()
    test_loss, test_acc = test()
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    test_loss_list.append(test_loss)
    test_acc_list.append(test_acc)
    print('Epoch: {}, train_loss: {:.6f}, train_acc: {:.2f}, test_loss: {:.6f}, test_acc: {:.2}'.format(i, train_loss, train_acc, test_loss, test_acc))

# test_x = torch.unsqueeze(test_data.data, dim = 1).type(torch.FloatTensor)[:10]/255.0
# test_output = model(test_x)
# pred = torch.max(test_output, 1)[1].data.numpy().squeeze()
# fig = plt.figure(figsize=(10, 4))
# img_number = 5
# for i in range(img_number):
#     ax = fig.add_subplot(2, 10/2, i+1)
#     ax.imshow(numpy.squeeze(test_x[i]), cmap = None)
#     ax.set_title('pred:' + str(pred[i].item()))

plt.show()
    