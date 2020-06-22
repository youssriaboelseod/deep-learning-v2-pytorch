import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms

import helper
import dog_catModel
# Define a transform to normalize the data

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5], 
                                                            [0.5, 0.5, 0.5])])
# Download and load the training data
trainset = datasets.ImageFolder(r'E:\Contact group_delete\deepLearning\Cat_Dog_data\Cat_Dog_data\train', transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.ImageFolder(r'E:\Contact group_delete\deepLearning\Cat_Dog_data\Cat_Dog_data\test', transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
image, label = next(iter(trainloader))
helper.imshow(image[0,:]);

# Create the network, define the criterion and optimizer

model = dog_catModel.Network(784, 10, [512, 256, 128])
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#dog_catModel.train(model, trainloader, testloader, criterion, optimizer, epochs=2)

#for save trained netowrk

#print("Our model: \n\n", model, '\n')
#print("The state dict keys: \n\n", model.state_dict().keys())

checkpoint = {'input_size': 784,
              'output_size': 10,
              'hidden_layers': [each.out_features for each in model.hidden_layers],
              'state_dict': model.state_dict()}

#torch.save(checkpoint, 'checkpoint.pth')


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = dog_catModel.Network(checkpoint['input_size'],
                             checkpoint['output_size'],
                             checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

model = load_checkpoint('checkpoint.pth')
print(model)