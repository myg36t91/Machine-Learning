from cProfile import label
from cv2 import readOpticalFlow
import torch
import numpy
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader 
from torch import nn, optim

data_array = numpy.arange(1, 721, 1)
predict_sin = numpy.sin(data_array * numpy.pi / 180) + numpy.random.randn(720) * 0.05

current_input = []
next_input = []

for i in range(0, 710):
    list = []
    for j in range(i, i+10):
        list.append(predict_sin[j])
    current_input.append(list)
    next_input.append(predict_sin[j+1])

current_input = numpy.array(current_input)
next_input = numpy.array(next_input)

train_data_array = current_input[:360]
train_predict_sin = next_input[:360]
test_data_array = current_input[360:]
test_predict_sin = next_input[360:]

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
        self.rnn = nn.RNN(input_size = 1, hidden_size = 5, num_layers = 1, batch_first = True)
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