import torch
import numpy
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader 
from torch import nn, optim

data_array = numpy.arange(0, 1000, 1, dtype = float)

for i in data_array:
    if(i<250):
        data_array[int(i)] = numpy.sin(i * numpy.pi / 25)
    elif(i>=250 and i<500):
        data_array[int(i)] = 1
    elif(i>=500 and i<750):
        data_array[int(i)] = -1
    else:
        data_array[int(i)] = 0.3 * numpy.sin( i * numpy.pi / 25) + 0.1 * numpy.sin( i * numpy.pi / 32) + 0.6 * numpy.sin( i * numpy.pi / 10)

output = data_array
data_array = numpy.arange(1, 1001, 1, dtype = float)

current_input = []
next_input = []

for i in range(0, 990):
    list = []
    for j in range(i, i+10):
        list.append(output[j])
    current_input.append(list)
    next_input.append(output[j+1])

current_input = numpy.array(current_input)
next_input = numpy.array(next_input)

train_data_array = current_input
train_predict_sin = next_input
test_data_array = current_input
test_predict_sin = next_input

class timeseries(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype = torch.float32)
        self.y = torch.tensor(y, dtype = torch.float32)
        self.len = x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.len

dataset = timeseries(train_data_array, train_predict_sin)
train_loader = DataLoader(dataset, shuffle = True, batch_size = 64)

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size = 1, hidden_size = 5, num_layers = 1, batch_first = True) # input_size transfrom to 1 dim
        self.linear = nn.Linear(5, 1)

    def forward(self, x):
        rnn_out, hidden_state = self.rnn(x)
        out = rnn_out[:, -1, :]
        out = self.linear(out)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RNN().to(device)
loss_function = nn.MSELoss()
optimator = optim.Adam(model.parameters(), lr = 1e-3)

epochs = 100
loss_array = []
for i in range(epochs):
    for j, data in enumerate(train_loader):
        pred = model(data[:][0].view(-1, 10, 1)).reshape(-1)
        loss = loss_function(pred, data[:][1])
        loss.backward()
        optimator.step()
        loss_array.append(loss.item())
    if i%10 == 0:
        print("epoch:{}, loss:{:.6f}".format(i, loss.item()))

test_set = timeseries(test_data_array, test_predict_sin)
test_pred = model(test_set[:][0].view(-1, 10, 1)).view(-1)

plt.figure(1)
plt.plot(loss_array)
plt.figure(2)
plt.plot(test_pred.detach().numpy(), label = 'pred')
plt.plot(test_set[:][1].view(-1), label = 'org')
plt.legend()
plt.show()