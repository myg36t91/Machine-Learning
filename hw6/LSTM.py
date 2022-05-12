import torch, pandas, numpy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from torch import nn, optim

data_file = pandas.read_csv("temperature.csv", header = 0)

scaler = MinMaxScaler(feature_range = (-1, 1))

class TemperatureDataset(Dataset):
    def __init__(self, data):
        self.data_file = data
        self.original_dataset = self.data_file.mean_temp.to_numpy()
        self.normalized_dataset = numpy.copy(self.original_dataset)
        self.normalized_dataset = self.normalized_dataset.reshape(-1, 1)
        self.normalized_dataset = scaler.fit_transform(self.normalized_dataset)
        self.normalized_dataset = self.normalized_dataset.reshape(-1)   
        self.sample_len = 12    
    
    def __len__(self):
        if len(self.original_dataset) > self.sample_len:
            return len(self.original_dataset) - self.sample_len
        else:
            return 0

    def __getitem__(self, idx):
        target = self.normalized_dataset[self.sample_len + idx]
        target = numpy.array(target).astype(numpy.float64)
        i = self.normalized_dataset[idx:(idx + self.sample_len)]
        i = i.reshape((-1, 1))
        i = torch.from_numpy(i)
        target = torch.from_numpy(target)
        return i, target

dataset = TemperatureDataset(data_file)
train_len = int(0.7 * len(dataset))
test_len = len(dataset) - train_len

train_data, test_data = random_split((dataset), [train_len, test_len])
train_loader = DataLoader(train_data, shuffle = False, batch_size = 32)
test_loader = DataLoader(test_data, shuffle = False, batch_size = 32)
        
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size = 1, hidden_size = 500, num_layers= 3, dropout = 0.1, batch_first = True)
        self.linear = nn.Linear(500, 1)

    def forward(self, x):
        hidden_0 = torch.zeros([3, x.shape[0], 500], dtype = torch.double)
        cell_0 = torch.zeros([3, x.shape[0], 500], dtype = torch.double)
        out, _ = self.lstm(x, (hidden_0.detach(), cell_0.detach()))
        out = self.linear(out[:, -1, :])
        return out

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LSTM()
model = model.double()
model = model.to(device)

loss_f = nn.MSELoss()
opt = optim.Adam(model.parameters(), lr = 1e-4)

# print(model)

def train():
    model.train()
    train_loss = 0

    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        opt.zero_grad()
        pred = model(data)
        pred = pred.view(-1)
        loss = loss_f(pred, target)
        loss.backward()
        opt.step()
        train_loss += loss

    print("train_loss: {:.6f}".format(train_loss))
    return train_loss

def test():
    model.eval()
    test_loss = 0

    for idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        pred = model(data)
        pred = pred.view(-1)
        loss = loss_f(pred, target)
        test_loss += loss

    print("test_loss: {:.6f}".format(test_loss))
    return test_loss

def predict(data):
    model.eval()
    with torch.no_grad():
        pred = model(data)
        return pred

for i in range(50):
    print("epoch = {}".format(i))
    train()
    test()

preds = []

for i in range(len(dataset)):
    normalized_temp, target = dataset[i]
    temp = normalized_temp
    temp = temp.view(1, 12, 1) 
    pred = predict(temp)
    act_pred = scaler.inverse_transform(pred.reshape(-1, 1))  
    preds.append(act_pred.item())

months = range(0, data_file.month.size)
mean_temp = data_file.mean_temp

plt.figure(figsize = (15, 5))
plt.title("Temperture Predict", fontsize = 16)
plt.plot(months, mean_temp, 'b', label = 'original')
plt.plot(months[12:], preds, 'r', label = 'predict')
plt.legend()
plt.show()