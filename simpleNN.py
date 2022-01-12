import torch
import torchvision
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()

        # layer 1
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = int(28*28)
    num_classes = 10
    learning_rate = .001
    batch_size = 64
    num_epochs = 3

    # Load Training and Test data
    train_dataset= datasets.MNIST(root="datasets/",
                                  train=True,
                                  transform= transforms.ToTensor(),
                                  download=True)

    test_dataset = datasets.MNIST(root="datasets",
                                  train=False,
                                  transform=transforms.ToTensor(),
                                  download= True)

    train_loader = DataLoader(dataset = train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(dataset = test_dataset,
                             batch_size= batch_size,
                             shuffle=True)

    # Initialize network
    model = NN(input_size, num_classes)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train Network
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            # get data to cuda for better performance
            data = data.to(device=device)
            targets = targets.to(device=device)

            # Reshape data
            data = data.reshape(data.shape[0], -1)

            # forward
            scores = model(data)

            # calculate Loss
            loss = criterion(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # Take Step for optimizer
            optimizer.step()


    def check_accuracy(loader, model):
        num_correct = 0
        num_samples = 0

        # Switch on eval mode for testing phase
        model.eval()

        # Turn off gradient tracking mode
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device = device)
                y = y.to(device = device)

                # reshape data
                x = x.reshape(x.shape[0], -1)

                # model prediction scores
                scores = model(x)

                # predictions
                _, predictions = scores.max(1)

                # Summing for True values
                num_correct += (predictions == y).sum()

                num_samples += predictions.size(0)

            model.train()
            return num_correct/num_samples

    print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")

    print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")
