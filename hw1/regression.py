import torch, pandas, numpy
import matplotlib.pyplot as plt
from torch import nn, optim

data_file = pandas.read_csv('pokemon.csv_path')

x = numpy.array(data_file['cp'])
y = numpy.array(data_file['cp_new'])

x = numpy.reshape(x, (x.size, 1))
y = numpy.reshape(y, (y.size, 1))

x = x.astype(numpy.float32)
y = y.astype(numpy.float32)

x = torch.tensor(x)
y = torch.tensor(y)

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        return x

model = Model(1, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.000001)
loss_list = []

epoch = 20
for i in range(epoch):
    lossfunction = nn.MSELoss()
    loss = lossfunction(model(x), y)
    loss.backward()
    opt.zero_grad()
    opt.step()
    loss_list.append(loss)
    print(loss.item())

plt.figure(1)
plt.plot(x, y, 'o', 'red')
plt.show()

plt.figure(1)
plt.plot(loss_list, 'r')

plt.figure(2)
plt.plot(x, y, 'o', 'r')
plt.plot(x, model(x).detach().numpy())
plt.show()