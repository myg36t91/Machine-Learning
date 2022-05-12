import torch, numpy, seaborn
import matplotlib.pyplot as plt
from cv2 import transform
from matplotlib import transforms
from torch.utils.data import Dataset, DataLoader
from torch import device, nn, optim
from torchvision import datasets
from torchvision.transforms import transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, )), ])

train_data = datasets.MNIST('Project/data/mnist', train = True, transform = transform, download = True)
test_data = datasets.MNIST('Project/data/mnist', train = False, transform = transform, download = True)

train_loader = DataLoader(train_data, batch_size = 64, shuffle = True, drop_last = True)
test_loader = DataLoader(test_data, batch_size = 64, shuffle = True)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 1, 4, 2, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.net(x)
        return x

class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 256, 4, 2, 0, bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 3, bias = False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.net(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
d_model = Discriminator().to(device)
g_model = Generator(100).to(device)
loss_function = nn.BCELoss()
d_model_opt = optim.Adam(d_model.parameters(), lr = 1e-5)
g_model_opt = optim.Adam(g_model.parameters(), lr = 1e-5)

d_model_loss = []
g_model_loss = []

for i in range(0, 1):
    for idx, (data, target) in enumerate(train_loader):
        # d_model
        d_model_opt.zero_grad()
        x_real = data.to(device)
        y_real = torch.ones(64, ).to(device)
        y_real_pred = d_model(x_real)
        d_real_loss = loss_function(y_real_pred.view(-1), y_real)
        d_real_loss.backward()
        # -------------------------------------------------------
        noise = torch.randn(64, 100, 1, 1, device = device)
        x_fake = g_model(noise)
        y_fake = torch.ones(64, ).to(device)
        y_fake_pred = d_model(x_fake)
        d_fake_loss = loss_function(y_fake_pred.view(-1), y_fake)
        d_fake_loss.backward()
        # -------------------------------------------------------
        d_model_loss.append(d_fake_loss.item() + d_real_loss.item())
        d_model_opt.step()

        # g_model
        g_model_opt.zero_grad()
        noise = torch.randn(64, 100, 1, 1, device = device)
        x_fake = g_model(noise)
        fake = torch.ones(64, ).to(device)
        fake_pred = d_model(x_fake)
        g_loss = loss_function(fake_pred.view(-1), fake)
        g_loss.backward()
        #--------------------------------------------------------
        g_model_loss.append(g_loss)
        g_model_opt.step()

        if idx % 100 == 0:
            with torch.no_grad():
                print("[{}/{}], d_model_loss : {:.6f}, g_model_loss : {:.6f}".format(i+1, 1, d_model_loss[idx], g_model_loss[idx]))

torch.save(g_model.state_dict(), 'Generator.pt')