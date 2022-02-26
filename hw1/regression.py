import torch, pandas, numpy, cv2
import matplotlib.pyplot as plt
from torch import nn, optim

# show cat.jpg using cv2
img = cv2.imread('D:\Downloads\hw1\data\cat.jpg')
cv2.imshow('images', img)
cv2.waitKey(0)
cv2.destroyAllWindows

# show cat.jpg using plt
img = plt.imread('D:\Downloads\hw1\data\cat.jpg')
plt.imshow(img)
plt.show()

# linear regression
data_file = pandas.read_csv('D:\Downloads\hw1\data\pokemon.csv')
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
    opt.zero_grad()
    lossfunction = nn.MSELoss()
    loss = lossfunction(model(x), y)
    loss.backward()
    opt.step()
    loss_list.append(loss)
    print(loss.item())

plt.figure(1)
plt.plot(loss_list, 'r')
plt.figure(2)
plt.plot(x, y, 'o', 'r')
plt.plot(x, model(x).detach().numpy())
plt.show()
