import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from  torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        # Average Pooling Layer
        self.pool = nn.AvgPool2d(kernel_size=(2, 2),
                                  stride=(2, 2),
                                  padding=0)
        # Relu
        self.relu = nn.ReLU()

        # Conv layers
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=6,
                               kernel_size=(5, 5),
                               stride=(1, 1),
                               padding=(0, 0))

        self.conv2 = nn.Conv2d(in_channels=6,
                               out_channels=16,
                               kernel_size=(5, 5),
                               stride=(1, 1),
                               padding=(0, 0))

        self.conv3 = nn.Conv2d(in_channels=16,
                               out_channels=120,
                               kernel_size=(5, 5),
                               stride=(1, 1),
                               padding=(0, 0))

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=120,
                             out_features=84)

        self.fc2 = nn.Linear(in_features= 84,
                             out_features= 10)


    def forward(self, x):
        """
        Forward Propogation
        """
        print(f"--{x.shape}")
        x = self.relu(self.conv1(x))
        print(f"--{x.shape}")
        x = self.pool(x)
        print(f"--{x.shape}")
        x = self.relu(self.conv2(x))
        print(f"--{x.shape}")
        x = self.pool(x)
        print(f"--{x.shape}")
        x = self.relu(self.conv3(x))

        # reshapping
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    """
    MAIN
    """
    # Constants
    BATCH = 64
    LEARNING_RATE= .001
    EPOCHS = 5

    # Device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_dataset = datasets.MNIST(root="./datasets/", train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root="./datasets/", train=False, transform=transforms.ToTensor(), download=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=True)

    # Initialize model
    model = LeNet().to(device)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train model
    for epoch in range(EPOCHS):

        # List to store the loss/error of each epoch
        losses = list()

        for bathc_idx, (data, target) in enumerate(tqdm(train_loader)):
            # Use cuda if possible
            data = data.to(device= device)
            target = target.to(device= device)

            # Forward
            scores = model(data)
            loss = criterion(scores, target)
            losses.append(loss.item())

            # Backprop
            optimizer.zero_grad()
            loss.backward()

            # Step for optimizer
            optimizer.step()

        print(f"===> Epoch: {epoch}\tLoss: {sum(loss)/len(losses):.5f}")



if __name__ == "__main__":
    main()
