
from matplotlib import image
import torch
from torch import device, nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision import datasets
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
G = Generator(100, 10).to(device)
G.load_state_dict(torch.load('cGenerator_hw.pt', map_location = torch.device('cpu')))

img = []
label_dim = 10
z_dim = 100

temp_noise = torch.randn (label_dim, z_dim)
fixed_noise = temp_noise
fixed_c = torch.zeros(label_dim, 1)

for i in range( 0 ,9):
    fixed_noise = torch.cat((fixed_noise, temp_noise), 0)
    temp = torch.ones (label_dim, 1) + i
    fixed_c = torch.cat((fixed_c, temp), 0)

fixed_noise = fixed_noise.view(-1, z_dim, 1, 1)
fixed_label = torch.zeros(100, label_dim)
fixed_label.scatter_(1, fixed_c.type(torch.LongTensor), 1)
fixed_label= fixed_label.view(-1, label_dim, 1, 1)

fake = G(fixed_noise, fixed_label)
img.append(make_grid(fake, padding = 0, normalize = True))

fig = plt.figure(dpi = 200)
im = img[0].cpu().detach().numpy().transpose(1, 2, 0)
plt.imshow(im)
plt.xticks([])
plt.yticks([])
plt.show()
