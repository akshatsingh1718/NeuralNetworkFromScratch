import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

class VGGNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(VGGNet, self).__init__()

        # pooling
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))

        # activation function
        self.act = nn.ReLU()

        ## convolution layers
        # conv 1
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv11 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # conv 2
        self.conv2 = nn.Conv2d(in_channels= 64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv22 = nn.Conv2d(in_channels= 128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # conv 3
        self.conv3 = nn.Conv2d(in_channels= 128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv33 = nn.Conv2d(in_channels= 256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # conv 4
        self.conv4 = nn.Conv2d(in_channels= 256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv44 = nn.Conv2d(in_channels= 512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        ## Fully connected layers
        # FC1
        self.fc1 = nn.Linear(in_features=512*7*7, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        # FC2
        self.fc3 = nn.Linear(in_features=4096, out_features= num_classes)


    def forward(self, x):
        # Layer 1
        x = self.act(self.conv1(x))
        x = self.act(self.conv11(x))
        x = self.pool(x)

        # Layer 2
        x = self.act(self.conv2(x))
        x = self.act(self.conv22(x))
        x = self.pool(x)

        # Layer 3
        x = self.act(self.conv3(x))
        x = self.act(self.conv33(x))
        x = self.act(self.conv33(x))
        x = self.pool(x)

        # Layer 4
        x = self.act(self.conv4(x))
        x = self.act(self.conv44(x))
        x = self.act(self.conv44(x))
        x = self.pool(x)

        # Layer 5
        x = self.act(self.conv44(x))
        x = self.act(self.conv44(x))
        x = self.act(self.conv44(x))
        x = self.pool(x)

        x= x.reshape(x.shape[0], -1)
        # Layer 6
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)

        return x


def main():
    # Global constants
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH = 64
    CHANNELS = 1
    EPOCHS = 5
    LEARNING_RATE= 0.001
    NUM_CLASSES= 10

    # LOAD DATA
    trasform= transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    train_dataset = datasets.MNIST(root="datasets/", train= True, transform= trasform, download=True)
    test_dataaset = datasets.MNIST(root="datasets/", train=False, transform= trasform, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(dataset=test_dataaset, batch_size=BATCH, shuffle=True)


    # Initialize network
    model = VGGNet(input_channels = CHANNELS, num_classes= NUM_CLASSES).to(device= DEVICE)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train model
    for epoch in range(EPOCHS):

      losses = list()
      for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        # Use cuda if possible
        data = data.to(device=DEVICE)
        target = target.to(device=DEVICE)

        # get predictions
        scores = model(data)
        loss = criterion(scores, target)

        # add loss to losses
        losses.append(loss.item())

        # backprop
        optimizer.zero_grad()
        loss.backward()

        # optimizer step
        optimizer.step()

      print(f"====> Epoch: {epoch}\tLoss: {sum(losses)/len(losses):.4f}")


if __name__ == "__main__":
    main()
