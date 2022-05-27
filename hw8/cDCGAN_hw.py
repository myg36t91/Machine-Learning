from calendar import EPOCH
from telnetlib import Telnet
from turtle import forward
from matplotlib import image
import torch
from torch import device, nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

class Discriminator(nn.Module):
    def __init__(self, c_dim, label_dim):
        super(Discriminator, self).__init__()
        self.input_x = nn.Sequential(
            nn.Conv2d(c_dim, 64, 4, 2, 1),
            nn.LeakyReLU()
        )
        self.input_y = nn.Sequential(
            nn.Conv2d(label_dim, 64, 4, 2, 1),
            nn.LeakyReLU()
        )
        self.concat = nn.Sequential(
            nn.Conv2d(64*2, 64, 4, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 1, 4, 2, 0),
            nn.Sigmoid() 
        )

    def forward(self, x, y):
        x = self.input_x(x)
        y = self.input_y(y)
        out = torch.cat([x, y], dim = 1)
        out = self.concat(out)
        return out

class Generator(nn.Module):
    def __init__(self, z_dim, label_dim):
        super(Generator, self).__init__()
        self.input_x = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 256, 4, 1, 0, bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.input_y = nn.Sequential(
            nn.ConvTranspose2d(label_dim, 256, 4, 1, 0, bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.concat = nn.Sequential(
            nn.ConvTranspose2d(256*2, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 3, bias = False),
            nn.Tanh()            
        )

    def forward(self, x, y):
        x = self.input_x(x)
        y = self.input_y(y)
        out = torch.cat([x, y], dim = 1)
        out = self.concat(out)
        return out

label_dim = 10
z_dim = 100

temp_noise = torch.randn( label_dim, z_dim)
fixed_noise = temp_noise
fixed_c = torch.zeros(label_dim, 1)

for i in range(0,9):
    fixed_noise = torch.cat((fixed_noise, temp_noise), 0)
    temp = torch.ones (label_dim, 1) +i
    fixed_c = torch.cat((fixed_c, temp), 0)

fixed_noise = fixed_noise.view(-1, z_dim, 1, 1)

# print('pred noise:', fixed_noise.shape)
# print('pred label.', fixed_c. shape, '\t', fixed_c[10])

fixed_label = torch.zeros(100, label_dim)
fixed_label.scatter_(1, fixed_c.type( torch.LongTensor), 1)
fixed_label = fixed_label.view(-1, label_dim, 1, 1)

# print('onehot label:', fixed_label.shape, '\t', fixed_label[10])

onehot = torch.zeros(label_dim, label_dim)
onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(label_dim, 1), 1).view(label_dim, label_dim, 1, 1)

image_size = 28

fill = torch.zeros([label_dim, label_dim, image_size, image_size])
for i in range(label_dim):
    fill[i, i, :, :] = 1

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, )), ])

batch_size = 64

train_data = datasets.FashionMNIST('Project/data/fashionmnist', train = True, transform = transform, download = True)
test_data = datasets.FashionMNIST('Project/data/fashionmnist', train = False, transform = transform, download = True)

train_loader = DataLoader(train_data, shuffle = True, batch_size = batch_size, drop_last = True)
test_loader = DataLoader(test_data, shuffle = False, batch_size = batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D = Discriminator(1, 10).to(device)
G = Generator(100, 10).to(device)
loss_f = nn.BCELoss()
opt_D = optim.Adam(D.parameters(), lr = 1e-5)
opt_G = optim.Adam(D.parameters(), lr = 1e-5)

D_avgloss = []
G_avgloss = []

epoch = 20

for i in range(epoch):
    D_loss = []
    G_loss = []

    for idx, (data, target) in enumerate (train_loader):
        opt_D.zero_grad()
        x_real = data.to(device)
        label = target.to(device)
        y_real = torch.ones(batch_size, ).to(device)
        c_real = fill[label].to(device)
        y_real_pred = D(x_real, c_real).squeeze()
        d_real_loss = loss_f(y_real_pred, y_real)
        d_real_loss.backward()

        noise = torch.randn(batch_size, z_dim, 1, 1, device = device)
        noise_lable = (torch.rand(batch_size, 1) * label_dim).type(torch.LongTensor).squeeze()
        noise_lable_onehot = onehot[noise_lable].to(device)

        x_fake = G(noise, noise_lable_onehot)
        y_fake = torch.zeros(batch_size, ).to(device)
        c_fake = fill[noise_lable].to(device)
        y_fake_pred = D(x_fake, c_fake).squeeze()
        d_fake_loss = loss_f(y_fake_pred, y_fake)
        d_fake_loss.backward()
        opt_D.step()

        opt_G.zero_grad()
        noise = torch.randn(batch_size, z_dim, 1, 1, device = device)
        noise_lable = (torch.rand(batch_size, 1) * label_dim).type(torch.LongTensor).squeeze()
        noise_lable_onehot = onehot[noise_lable].to(device)
        x_fake = G(noise, noise_lable_onehot)
        c_fake = fill[noise_lable].to(device)
        y_fake_pred = D(x_fake, c_fake).squeeze()
        g_loss = loss_f(y_fake_pred, y_real)
        g_loss.backward()
        opt_G.step()

        D_loss.append(d_fake_loss.item() + d_real_loss.item())
        G_loss.append(g_loss.item())

        if idx % (int(len(train_loader) / 100)) == 0:
            with torch.no_grad():
                print("Epoch[{:02}/{}] \t stpe[{:05}/{}] \t D_loss : {:.6f} \t G_loss : {:.6f}".format(i+1, epoch, idx+1, len(train_loader), D_loss[idx], G_loss[idx]))

D_avgloss.append(torch.mean(torch.FloatTensor(D_loss)))
G_avgloss.append(torch.mean(torch.FloatTensor(G_loss)))

torch.save(G.state_dict(), 'cGenerator_hw.pt')

plt.figure()
plt.plot(D_avgloss, label = "D_loss")
plt.plot(G_avgloss, label = "G_loss")
plt.legend()
plt.show()