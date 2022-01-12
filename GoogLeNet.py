# imports
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm


# Conv class
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__()
        # activation function
        self.act = nn.ReLU()
        # batch norm
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)
        # conv
        self.conv = nn.Conv2d(in_channels= in_channels, out_channels=out_channels, **kwargs)

    def forward(self, x):
        return self.act(self.batch_norm(self.conv(x)))


# Inseption class
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, conv1x1_out, conv3x3_in, conv3x3_out, conv5x5_in, conv5x5_out, pool1x1_out):
        super(InceptionBlock, self).__init__()
        # conv 1
        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=conv1x1_out, kernel_size=(1, 1))

        # conv 2
        self.conv2 = nn.Sequential(
            ConvBlock(in_channels= in_channels, out_channels=conv3x3_in, kernel_size=(1, 1)),
            ConvBlock(in_channels= conv3x3_in, out_channels=conv3x3_out, kernel_size=(3, 3), padding=(1, 1)),
        )

        # conv 3
        self.conv3 = nn.Sequential(
            ConvBlock(in_channels= in_channels, out_channels=conv5x5_in, kernel_size=(1, 1)),
            ConvBlock(in_channels= conv5x5_in, out_channels= conv5x5_out, kernel_size=(5, 5), padding=(2, 2))
        )

        # conv 4
        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ConvBlock(in_channels= in_channels, out_channels=pool1x1_out, kernel_size=(1, 1), stride=(1, 1))
        )

    def forward(self, x):
        # print("-----", self.conv1(x).shape)
        # print("-----", self.conv2(x).shape)
        # print("-----", self.conv3(x).shape)
        # print("-----", self.conv4(x).shape)


        x = torch.cat([
            self.conv1(x),
            self.conv2(x),
            self.conv3(x),
            self.conv4(x)
        ], 1)
        return x


# GoogLeNet class
class GoogLeNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(GoogLeNet, self).__init__()
        # conv 1
        self. conv1 = ConvBlock(in_channels= in_channels, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))

        # max pool
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # conv 2
        self.conv2 = ConvBlock(64, 192, kernel_size=3, stride=1, padding= 1)

        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Inception Block 1
        self.inception3a = InceptionBlock(in_channels= 192, conv1x1_out=64, conv3x3_in=96, conv3x3_out=128, conv5x5_in=16, conv5x5_out=32, pool1x1_out=32)
        self.inception3b = InceptionBlock(in_channels= 256, conv1x1_out=128, conv3x3_in=128, conv3x3_out=192, conv5x5_in=32, conv5x5_out=96, pool1x1_out=64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Inception Block 2
        self.inception4a = InceptionBlock(in_channels=480, conv1x1_out=192, conv3x3_in=96, conv3x3_out=208, conv5x5_in=16, conv5x5_out=48, pool1x1_out=64)
        self.inception4b = InceptionBlock(in_channels=512, conv1x1_out=160, conv3x3_in=112, conv3x3_out=224, conv5x5_in=24, conv5x5_out=64, pool1x1_out=64)
        self.inception4c = InceptionBlock(in_channels=512, conv1x1_out=128, conv3x3_in=128, conv3x3_out=256, conv5x5_in=24, conv5x5_out=64, pool1x1_out=64)
        self.inception4d = InceptionBlock(in_channels=512, conv1x1_out=112, conv3x3_in=114, conv3x3_out=288, conv5x5_in=32, conv5x5_out=64, pool1x1_out=64)
        self.inception4e = InceptionBlock(in_channels=528, conv1x1_out=256, conv3x3_in=160, conv3x3_out=320, conv5x5_in=32, conv5x5_out=128, pool1x1_out=128)

        # max pool 4
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Inception Block 3
        self.inception5a = InceptionBlock(in_channels=832, conv1x1_out=256, conv3x3_in=160, conv3x3_out=320, conv5x5_in=32, conv5x5_out=128, pool1x1_out=128)
        self.inception5b = InceptionBlock(in_channels=832, conv1x1_out=384, conv3x3_in=192, conv3x3_out=384, conv5x5_in=48, conv5x5_out=128, pool1x1_out=128)

        # avg pool1
        self.avgpool1 = nn.AvgPool2d(kernel_size=7, stride=1)

        # droupout
        self.droupout = nn.Dropout(p=.4)

        # Linear layers
        self.fc1 = nn.Linear(in_features=1024, out_features= num_classes)

        # SOFTMAX
        self.softmax = nn.Softmax()


    def forward(self, x):
        # Conv 1
        x = self.conv1(x)
        print('conv1---------', x.shape)


        # Max pool 1
        x = self.maxpool1(x)
        print('Max pool 1---------', x.shape)


        # Conv 2
        x = self.conv2(x)
        print('Conv 2---------', x.shape)


        # Max pool 2
        x = self.maxpool2(x)
        print('Max pool 2---------', x.shape)


        # Inception 3
        x = self.inception3a(x)
        print('Inception 3a---------', x.shape)
        x = self.inception3b(x)
        print('Inception 3b---------', x.shape)
        x = self.maxpool3(x)
        print('Max Pool 3---------', x.shape)


        # Inception 4
        x = self.inception4a(x)
        print('Inception 4a---------', x.shape)
        x = self.inception4b(x)
        print('Inception 4b---------', x.shape)
        x = self.inception4c(x)
        print('Inception 4c---------', x.shape)
        x = self.inception4d(x)
        print('Inception 4d---------', x.shape)
        x = self.inception4e(x)
        print('Inception 4e---------', x.shape)

        # Maxpool
        x = self.maxpool4(x)
        print('Max pool 4---------', x.shape)


        # Inception 5
        x = self.inception5a(x)
        print('Inception 5a---------', x.shape)
        x = self.inception5b(x)
        print('Inception 5b---------', x.shape)

        # Average pool
        x = self.avgpool1(x)
        print('Average pool---------', x.shape)


        # Reshapping
        x = x.reshape(x.shape[0], -1)

        # droupout
        x = self.droupout(x)

        # linear layers
        x = self.fc1(x)

        return self.softmax(x)



def main():
    # Global constants
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH = 64
    CHANNELS = 1
    EPOCHS = 3
    LEARNING_RATE= 0.001
    NUM_CLASSES= 10

    # LOAD DATA
    trasform= transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    train_dataset = datasets.MNIST(root="datasets/", train= True, transform= trasform, download=True)
    test_dataaset = datasets.MNIST(root="datasets/", train=False, transform= trasform, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(dataset=test_dataaset, batch_size=BATCH, shuffle=True)


    # Initialize network
    model = GoogLeNet(in_channels = CHANNELS, num_classes= NUM_CLASSES).to(device= DEVICE)

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
