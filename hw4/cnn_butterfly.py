import torch, pandas, numpy
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from PIL import Image

train_path = 'hw4/data/butterfly/train'
test_path = 'hw4/data/butterfly/test'

train_transform = transforms.Compose([transforms.Resize((25, 25)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_transform = transforms.Compose([transforms.Resize((25, 25)), transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])

train_data = datasets.ImageFolder(train_path, transform = train_transform)
test_data = datasets.ImageFolder(test_path, transform = test_transform)

train_loader = DataLoader(train_data, batch_size = 64, shuffle = True)
test_loader = DataLoader(test_data, batch_size = 64, shuffle = True)

print('label:', train_data.class_to_idx)
print('path & label', train_data.imgs[0])
print('image')
print(train_data[0][0])
print(train_data[0][1])
print('size:', train_data[0][0].shape)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer1 = nn.Conv2d(3, 32, 3, 1) #23
        self.relu_layer1 = nn.ReLU()
        self.maxpool_layer1 = nn.MaxPool2d(2) #11
        self.conv_layer2 = nn.Conv2d(32, 64, 3, 1) #9
        self.relu_layer2 = nn.ReLU()
        self.maxpool_layer2 = nn.MaxPool2d(2) #4
        # self.conv_layer3 = nn.Conv2d(64, 128, 3, 1) #21
        # self.relu_layer3 = nn.ReLU()
        # self.maxpool_layer3 = nn.MaxPool2d(2) #10
        self.function1 = nn.Linear(64*4*4, 512)
        self.relu_layer4 = nn.ReLU()
        self.function2 = nn.Linear(512, 75)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.relu_layer1(x)
        x = self.maxpool_layer1(x)
        x = self.conv_layer2(x)
        x = self.relu_layer2(x)
        x = self.maxpool_layer2(x)
        # x = self.conv_layer3(x)
        # x = self.relu_layer3(x)
        # x = self.maxpool_layer3(x)
        x = x.view(-1, 64*4*4)
        x = self.function1(x)
        x = self.relu_layer4(x)
        x = self.function2(x)
        x = self.sigmoid(x)
        return x

device = torch.device('cpu')
model = CNN()
lossFunction = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr= 1e-5)

def train():
    model.train()
    train_loss = 0
    # train_acc = 0
    for index, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        pred = model(data)
        loss = lossFunction(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    return train_loss / len(train_loader)

def test():
    model.eval()
    test_loss = 0
    # test_acc = 0
    for index, (data, label) in enumerate(test_loader):
        data, label = data.to(device), label.to(device)
        pred = model(data)
        loss = lossFunction(pred, label)
        test_loss += loss.item()
    
    return test_loss / len(test_loader)

trainLossList = []
testLossList = []
# epoch = 2
# for i in range(epoch):
#     train_loss = train()
#     test_loss = test()
#     trainLossList.append(train_loss)
#     testLossList.append(test_loss)
#     print(('epoch: {}, train_loss:{:.6f}, test_loss:{:.6f}').format(i, train_loss, test_loss))

# torch.save(model.state_dict(), 'CNN.pt')
# df = pandas.DataFrame((trainLossList, testLossList))
# df = df.T
# df.to_csv('loss.csv', header = 0, index = 0)

model.load_state_dict(torch.load('CNN.pt'))
model.eval()
img = Image.open('hw4/data/butterfly/6 images/1.jpg').convert('RGB')
data = train_transform(img)
data = torch.unsqueeze(data, dim = 0)
pred = model(data)
_, y = torch.max(pred, 1)
df = pandas.read_csv('loss.csv', header = None)
train_loss = numpy.array(df[0])
test_loss = numpy.array(df[1])

plt.figure(1)
plt.imshow(img)
# plt.title('cat' if y.cpu().numpy() == 0 else 'dog')

plt.figure(2)
plt.plot(train_loss)
plt.plot(test_loss)
plt.show()
