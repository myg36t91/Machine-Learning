import torch, numpy, pandas
import torch.utils.data as Data
import matplotlib.pyplot as plt
from torch import nn, optim

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear_layer1 = nn.Linear(input_dim, 500)
        self.linear_layer2 = nn.Linear(500, 250)
        self.linear_layer3 = nn.Linear(250, 125)
        self.linear_layer4 = nn.Linear(125, output_dim)
        self.fc = nn.Sigmoid()
    
    def forward(self, x):
        x = self.linear_layer1(x)
        x = self.linear_layer2(x)
        x = self.linear_layer3(x)
        x = self.linear_layer4(x)
        x = self.fc(x)
        return x

datafile = pandas.read_fwf('ml\hw2\data\579.txt', header=None)
x = numpy.array(datafile.iloc[:, 0:2], dtype='float32')
y = pandas.get_dummies(datafile[2]).to_numpy(dtype = 'float32')

x = torch.from_numpy(x)
y = torch.from_numpy(y)

train_data = Data.TensorDataset(x, y)
train_loader = Data.DataLoader(train_data, batch_size=1, shuffle=True)

model = MLP(2, 3)
lossfunction = nn.MSELoss()
opt = optim.Adam(model.parameters(), lr = 1e-6)

loss_list = []
acc_list = []

# training 50 time
epoch =  20
for i in range(epoch):
    train_loss = 0
    train_acc = 0
    for data, label in train_loader:
        pred = model(data)
        loss = lossfunction(pred, label)
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_loss += loss
        _, y = pred.max(1)
        correct = (y == label.max(1)[1]).sum().item()
        acc = correct / data.shape[0]
        train_acc += acc
    loss_list.append(train_loss)
    acc_list.append(train_acc)
    print('Epoch: {}, train_loss: {:.6f}, train_acc: {:.6f}'.format(i, train_loss/len(train_loader), train_acc/len(train_loader)))

fig = plt.figure()
y_axis = fig.add_subplot(111)
y_axis.plot(loss_list, 'r')
y_axis.set_ylabel('loss')
y_axis.legend(['train_loss'], loc='upper right')
x_axis = y_axis.twinx()
x_axis.plot(acc_list, 'b')
x_axis.set_xlabel('acc')
x_axis.legend(['train_acc'], loc='lower right')
plt.show()