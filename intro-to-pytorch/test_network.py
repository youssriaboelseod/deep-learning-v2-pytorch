import torch
from torchvision import datasets, transforms
import helper
import matplotlib.pyplot as plt

##step1- Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
#Here we can see one of the images.
image, label = next(iter(trainloader))
helper.imshow(image[0,:])

print(type(image))
print(image.shape)
print(label.shape)

plt.imshow(image[1].numpy().squeeze(), cmap='Greys_r');
##step2-create nural network
from torch import nn
import torch.nn.functional as F
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Defining the layers, 128, 64, 10 units each
        self.fc1 = nn.Linear(784, 250)
        self.fc2 = nn.Linear(250, 128)
        # Output layer, 10 units - one for each digit
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        
    def forward(self, x):
                # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        
        ''' Forward pass through the network, returns the output logits '''
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        x = F.log_softmax(self.fc4(x), dim=1)


        return x

model = Network()
#step3 calculate the loss
# Define the loss
criterion = nn.CrossEntropyLoss()
print("______________test___________")
print(criterion)
# Get our data

# Flatten images

# Forward pass, get our logits

#ster 4 train network
from torch import optim

optimizer = optim.SGD(model.parameters(), lr=0.003)

epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
    
        # TODO: Training pass
        optimizer.zero_grad()
        
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")

