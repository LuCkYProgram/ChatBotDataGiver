import torch
import torch.nn as nn
import torch.utils.data as td

# Set random seed for reproducability
torch.manual_seed(0)

print("Libraries imported - ready to use PyTorch", torch.__version__)

def prepare_data_pytorch(x_train, y_train, x_test, y_test):
    train_x = torch.Tensor(x_train).float()
    train_y = torch.Tensor(y_train).long()
    train_ds = td.TensorDataset(train_x,train_y)
    train_loader = td.DataLoader(train_ds, batch_size=20,
        shuffle=False, num_workers=1)

    test_x = torch.Tensor(x_test).float()
    test_y = torch.Tensor(y_test).long()
    test_ds = td.TensorDataset(test_x,test_y)
    test_loader = td.DataLoader(test_ds, batch_size=20,
        shuffle=False, num_workers=1)
    print('Ready to load data')
    hl = input("Input the number of hidden layer nodes: ")
    return( test_loader, train_loader, hl)


# Define the neural network
class PenguinNet(nn.Module):
    def __init__(self):
        super(PenguinNet, self).__init__()
        self.fc1 = nn.Linear(len(features), hl)
        self.fc2 = nn.Linear(hl, hl)
        self.fc3 = nn.Linear(hl, len(classes))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x

model = PenguinNet()
print(model)