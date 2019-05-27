# 1 - Welcome!
Learn to use PyTorch for deep learning models
  - basic intro (covering tensors)
  - autograd module calculates gradients for training neural networks
  - build network and run data forward through it
    - define loss and optimization method
    - generalize through validation
  - use pre-trained networks to improve performance of your classifier through transfer learning

# 2 - Pre-Notebook
Work through the notebooks
  - can check solutions
  - be sure to type the code (and understand it) for yourself

# 3 - Notebook Workspace
This is the notebook you'll be working in.

# 4 - Single Layer Neural Networks
Fundamental data structure for neural netwoks are tensors
  - vector is 1d tensor
  - matrix is 2d tensor
  - array with 3 indices is 3d tensor

Given input tensor, weight tensor and bias, calculate output of the network


# 5 - Single Layer Neural Networks: Solution
We'll want to do matrix multiplication and sum it (`torch.mm()` or `torch.matmul()`)
If your tensors aren't the right size/shape, you'll get a `size mismatch` error
  - use `tensor.shape` to get sahpe of a tensor
  - to reshape a tensor:
    - `tensor.reshape(a,b)` will give size a,b
    - `tensor.resize_(a,b)` returns same tensor with different shape (could result in lost elements depending on the shape)
    - `tensor.vew(a,b)` will give new tensor with same data but new size

```
# reshape the weights matrix because it is the wrong shape for multiplication
new_weights = weights.view(5,1)
# multiple features by weights and add in the bias
y = activation(torch.matmul(features, new_weights) + bias)
print(y)
```
  - this calculates the output for a single neuron
  - stack them so the output of one layer of neurons becomes the input for another layer
# 6 - Networks Using Matrix Multiplication

# 7 - Multilayer Networks Solution
```
hidden_layer = activation(torch.matmul(features, W1) + B1)
output_layer = activation(torch.matmul(hidden_layer, W2) + B2)
print(output_layer)
```
To get Hidden Layer:
  - Do a matrix multiplication between the initial feature layer and the first set of weights
    - run that + the bias through the activation function
To get Output Layer
  - Do a matrix multiplication between the  hidden layer and the second set of weights
    - run that + bias through activation function

Convert between NumPy arrays and Torch tensors
  - `torch.from_numpy()`
  - `my_tensor.numpy()`

The memory is shared between the numpy array and torch tensor so if you change the values in-place of one object the other will change

# 8 - Neural Networks in PyTorch
Deep Learning networks can have hundreds of layers
  - can be built with just weights and matrices like we've been doing
  - PyTorch module `nn` has classes/methods for efficiently building large neural networks

Torchvision package
  - has MNIST dataset
    - series of handdrawn digits with their appropriate label

Fully Connected / Dense Neural Networks
  - every unit in one layer is connected to each unit in the next layer
  - input to each layer must be 1d vector
  - input image is currently 28 x 28 but we need to flatten it to 784

```
## Solution
def activation(x):
    return 1/(1+torch.exp(-x))

# NOTE: Could use -1 as the final number (instead of 784) as it is shorthand to tell the system to figure out the shape on its own
# Flatten the input images
inputs = images.view(images.shape[0], 784)

# Create parameters
w1 = torch.randn(784, 256)
b1 = torch.randn(256)

w2 = torch.randn(256, 10)
b2 = torch.randn(10)

hidden_layer = activation(torch.matmul(inputs, w1) + b1)
output_layer = torch.matmul(hidden_layer, w2) + b2
print(output_layer)
```

This gives us 10 outputs for our network
  - we want to pass image to network and get probability over the classes that tells us the likely class the image belongs to

Run an image through the network and it will give you an output for each class
  - you'll want to run this through softmax
  - SoftMax squishes each input xi between 0 and 1 and normalizes the values to give a proper robability distribution (all outputs sum up to 1)

  sig(xi)= e^xi/sumKk e^xk

```
def softmax(x):
    return torch.exp(x)/torch.sum(torch.exp(x), dim=1).view(-1, 1)

#Above, we're reshaping with view, -1 is shorthand telling the computer to figure it out based on the second value which is a dimension of 1

# Here, out should be the output of the network in the previous excercise with shape (64,10)
probabilities = softmax(out)

# Does it have the right shape? Should be (64, 10)
print(probabilities.shape)
# Does it sum to 1?
print(probabilities.sum(dim=1))
```

Code for building a neural network using pytorch:
```
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10)
        
        # Define sigmoid activation and softmax output 
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        
        return x
```
  - automatically creates the weights and biases for the network
  - you can access them using `net.hidden.weight()` and `net.hidden.bias()`

Another way to define the network:
```
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10)
        
    def forward(self, x):
        # Hidden layer with sigmoid activation
        x = F.sigmoid(self.hidden(x))
        # Output layer with softmax activation
        x = F.softmax(self.output(x), dim=1)
        
        return x
```

Any function can be used as the activation function
  - for network to approximate a non-linear function the activation unctions must be non linear
  - SoftMax, Sigmoid, TanH and ReLU are common ones
    - ReLU is used almost excluseively as activation function for hidden layers

# 9 - Neural Networks Solution
```
def activate(x):
  return 1/(1+torch.exp(-x))

inputs = images.view(images.shape[0], -1)


# 10 - Implementing Softmax Solution
```
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784,128)
        self.hidden2 = nn.Linear(128,64)
        self.output = nn.Linear(64,10)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.softmax(self.output(x), dim=1)
        
        return x
            
model = Network()
model
```

Changes they made:
```
# Set biases to all zeros
model.hidden1.bias.data.fill_(0)

# sample from random normal with standard dev = 0.01
model.hidden1.weight.data.normal_(std=0.01)
```

Their run of a forward pass:
```
# Grab some data 
dataiter = iter(trainloader)
images, labels = dataiter.next()

# Resize images into a 1D vector, new shape is (batch size, color channels, image pixels) 
images.resize_(64, 1, 784)
# or images.resize_(images.shape[0], 1, 784) to automatically get batch size

# Forward pass through the network
img_idx = 0
ps = model.forward(images[img_idx,:])

img = images[img_idx]
helper.view_classify(img.view(1, 28, 28), ps)
```

Output is fairly random because network hasn't been trained

Using `nn.Sequential`
  use his to build an equivalent network:

```
# Hyperparameters for our network
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

# Build a feed-forward network
```
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.Softmax(dim=1))
print(model)

# Forward pass through the network and display output
images, labels = next(iter(trainloader))
images.resize_(images.shape[0], 1, 784)
ps = model.forward(images[0,:])
helper.view_classify(images[0].view(1, 28, 28), ps)
```

If you want to name the individual layers/operators you can pass in an `OrderedDict`

```
from collections import OrderedDict
model = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                      ('relu1', nn.ReLU()),
                      ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
                      ('relu2', nn.ReLU()),
                      ('output', nn.Linear(hidden_sizes[1], output_size)),
                      ('softmax', nn.Softmax(dim=1))]))
model
```

This will allow you to access layers by integer `model[0]` or by name `model.fc1`

# 11 - Network Architectures in PyTorch
Our network above is naive and doesn't know the function mapping the inputs to the outputs
  - train it by showing it examples of real data then adjusting network parameters so that it approximates the function
  - for any function we'll have desired input (number 4) and desired output (node for 4 has highest probability)

If you use non-linear activations and you have correct, labeled data, you can pass in an image and correct output and eventually your neural network will approximate this function

Need a Loss Function/Cost
  - measures our prediction error
  - we're going to use the Mean Squared Error
    l = 1/2n* nSUMi(yi - y-hati)2
    n is number of training examples, yi are true labels and y-hati are predicted labels
  - then we can adjust our weights to minimize this loss

Gradient Descent
  - slope of the loss function with respect to our parameters
  - points in the direction of fastest change

With multilayered neural networks we use Backpropagation to do this
  - application of the chain rule from calculus

Forward pass is what we've been doing
  - calculate the loss of the output

Backpropagation
  - popagate the gradient of the loss backwards through the network
  - each operation has some gradient between inputs and outputs
  - multiply incoming gradient with gradient for operation as we move backwards
  - use chainrule to calculate gradient of loss with resepct to the weights
  - update weights with this gradient and some learning rate alpha
  - NOTE: gradient helps us maximize our loss so to minimize, we can subtract the gradient from our weights

Losses in PyTorch
  - cross entropy (nn.CrossEntropyLoss)
  - loss is usually assigned to `criterion`
  - with softmax output you want to use cross-entropy as the loss
  - to calculate:
    - first define criterian
    - pass in output of your network and correct labels
  - NOTE about `nn.CrossEntropyLoss`
    - criterion combines `nn.LogSoftMax()` and `nn.NLLLoss()` in one class
    - input is expected to contain scores for each class
    * means we need to pass in raw output [called logits/scores] of our network to the loss (not the output of the softmax function)

```
# Build a feed-forward network
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10))

# Define the loss
criterion = nn.CrossEntropyLoss()

# Get our data
images, labels = next(iter(trainloader))
# Flatten images
images = images.view(images.shape[0], -1)

# Forward pass, get our logits
logits = model(images)
# Calculate the loss with the logits and the labels
loss = criterion(logits, labels)

print(loss)
```

Note:
  - teacher prefers models with log-softmax output using `nn.LogSoftmax` or `F.log_softmax`
  - then get actual probabilities by taking exponential `torch.exp(output)`
  - with log-softmax output you want to use negative log likelihood loss `nn.NLLLoss`

# 12 - Network Architecture: Solution
```
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

# TODO: Define the loss
criterion = nn.NLLLoss()

### Run this to check your work
# Get our data
images, labels = next(iter(trainloader))
# Flatten images
images = images.view(images.shape[0], -1)

# Forward pass, get our logits
logits = model(images)
# Calculate the loss with the logits and the labels
loss = criterion(logits, labels)

print(loss)
```

How do you use that loss to perform backpropagation
  - Torch module `autograd`!

Autograd
  - keeps track of operations performed on tensors
  - goes backward through those operations to calculate gradients
  - set `requires_grad = True` on tensor to make sure it's keeping track
  - can do that at creation with `requires_grad` or at any time with `x.requires_grad_(True)`
  - turn it off for a block of code with `torch.no_grad()`
  - turn it on or off altogether with ``torch.set_grad_enabled(True|False)``

Loss and Autograd Together
  - when we create a network with PyTorch, all parameters are intiialized with `requires_grad = True`
  - create loss then call `loss.backward()` gradients for the parameters are calculated
  - gradients are used to update the weigths with gradient descent

```
# Build a feed-forward network
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

logits = model(images)
loss = criterion(logits, labels)

print('Before backward pass: \n', model[0].weight.grad)

loss.backward()

print('After backward pass: \n', model[0].weight.grad)
```

Training the Network!
  - need an optimizer to use to update the weights with the gradients
  - this comes from PyTorch's `optim` package (e.g.Stochastic Gradient Descent with `optim.SGD`)

```
from torch import optim

# Optimizers require the parameters to optimize and a learning rate
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

General process with PyTorch:
 - Make a forward pass through the network
 - Use the network output to calculate the loss
 - Perform a backward pass through the network with loss.backward() to calculate the gradients
 - Take a step with the optimizer to update the weights

NOTE: When you do multiple backwards passes with the same parameters, the gradients are accumulated. Zero the gradients on each training pass with `optiizer.zero_grad()` or you'll retain previous gradients.

```
print('Initial weights - ', model[0].weight)

images, labels = next(iter(trainloader))
images.resize_(64, 784)

# Clear the gradients, do this because gradients are accumulated
optimizer.zero_grad()

# Forward pass, then backward pass, then update weights
output = model.forward(images)
loss = criterion(output, labels)
loss.backward()
print('Gradient -', model[0].weight.grad)

# Take an update step and few the new weights
optimizer.step()
print('Updated weights - ', model[0].weight)
```

# 13 - Training a Network: Solution
```
## Your solution here

model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)

epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
    
        optimizer.zero_grad()
        output = model.forward(images)
        
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
                
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")
```

# 14 - Classifying Fashion-MNIST
Train a neural network using the Fashion-MNIST dataset
  - set of 28x28 greyscale images of clothes

Code to load the data:

```
import torch
from torchvision import datasets, transforms
import helper

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
```

Define the model
```
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms

model = nn.Sequential(
  nn.Linear(784, 128),
  nn.ReLU(),
  nn.Linear(128, 64),
  nn.ReLU(),
  nn.Linear(64, 10),
  nn.LogSoftmax(dim=1))
```

Define criterion and optimizer:
```
from torch import optim

criterion = nn.NLLLoss()
optiizer = optim.SGD(model.parameters(), lr=0.01)
```

Train the model:
```
epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        images = images.view(images.shape[0], -1)
        optimizer.zero_grad()
        output = model.forward(images)
        
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")
```

Testing on some test data:
```
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import helper

# Test out your network!

dataiter = iter(testloader)
images, labels = dataiter.next()
img = images[0]
# Convert 2D image to 1D vector
img = img.resize_(1, 784)

with torch.no_grad():
    logps = model.forward(img)

# TODO: Calculate the class probabilities (softmax) for img
ps = torch.exp(logps)

# Plot the image and probabilities
helper.view_classify(img.resize_(1, 28, 28), ps, version='Fashion')
```

# 15 - Fashion-MNIST: Solution
My solution is posted above and works!

# 16 - Inference and Validation
Inference
  - user your trained network to make predictions
  - protect against overfitting using a test/validation set

When you're loading your data if you set `train=False` you'll get your test data instead of your training data

With the probabilities, we can get most likely class using `ps.topk` which returns the k highest values

```
top_p, top_class = ps.topk(1, dim=1)
# Look at the most likely classes for the first 10 examples
print(top_class[:10,:])
```

Response for that looks like this:
```
tensor([[ 1],
        [ 1],
        [ 1],
        [ 1],
        [ 1],
        [ 1],
        [ 1],
        [ 1],
        [ 1],
        [ 1]])
```

Now you can compare top_class and labels (but make sure to pay attention to the shape of your data)
  - top_class is 2D tensor (64,1) and labels is 1D tensor (64)
  - must have same shape for equality check to work as intended
  - `equals = top_class == labels.view(*top_class.shape)`

Calculate accuracy of the predictions on the testing data:
```
accuracy = torch.mean(equals.type(torch.FloatTensor))
print(f'Accuracy: {accuracy.item()*100}%')
```

Dropout
  - add this in to the model to have nodes drop out during training to help prevent over fitting
  - `self.dropout = nn.Dropout(p=0.2)` that goes in the model `init` function
  - modifies the function setup:
  ```
  # Now with dropout
  x = self.dropout(F.relu(self.fc1(x)))
  x = self.dropout(F.relu(self.fc2(x)))
  x = self.dropout(F.relu(self.fc3(x)))
  ```

General pattern for the evaluation loop:
```
# turn off gradients
with torch.no_grad():

    # set model to evaluation mode
    model.eval()

    # validation pass here
    for images, labels in testloader:
        ...

# set model back to train mode
model.train()
```

# 17 - Validation: Solution
Model with dropout:

```
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        # Now with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))

        # output so no dropout here
        x = F.log_softmax(self.fc4(x), dim=1)

        return x
```

# 18 - Dropout: Solution
Training with dropout:

```
model = Classifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

epochs = 30
steps = 0

train_losses, test_losses = [], []
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        
        optimizer.zero_grad()
        
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    else:
        test_loss = 0
        accuracy = 0
        
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            model.eval()
            for images, labels in testloader:
                log_ps = model(images)
                test_loss += criterion(log_ps, labels)
                
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        
        model.train()
        
        train_losses.append(running_loss/len(trainloader))
        test_losses.append(test_loss/len(testloader))

        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
              "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
              "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
```

Update display mode to turn off eval and autograd:
```
# Import helper module (should be in the repo)
import helper

# Test out your network!

model.eval()

dataiter = iter(testloader)
images, labels = dataiter.next()
img = images[0]
# Convert 2D image to 1D vector
img = img.view(1, 784)

# Calculate the class probabilities (softmax) for img
with torch.no_grad():
    output = model.forward(img)

ps = torch.exp(output)

# Plot the image and probabilities
helper.view_classify(img.view(1, 28, 28), ps, version='Fashion')
```

This is handy for seeing the shape of your training loss vs your validation loss

```
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Validation Loss')
plt.legend(frameon=False)
```

# 19 - Saving and Loading Models
You don't want to have to retrain a model every time you use it again

Simplest thing to do is save the state dict with `torch.save`
  - `torch.save(model.state_dict(), 'checkpoint.pth')`
Then you can load with `torch.load`
  - `state_dict = torch.load('checkpoint.pth')
print(state_dict.keys())`
And to load the state dict into the network you do the following:
  - `model.load_state_dict(state_dict)`

NOTE: This will only work if the model architecture is exactly the same as the checkpoint architecture
  - if not you get an error about sizes and shapes

Better Solution!
```
checkpoint = {'input_size': 784,
              'output_size': 10,
              'hidden_layers': [each.out_features for each in model.hidden_layers],
              'state_dict': model.state_dict()}

torch.save(checkpoint, 'checkpoint.pth')
```

You can write a function to load:
```
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = fc_model.Network(checkpoint['input_size'],
                             checkpoint['output_size'],
                             checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])
    
    return model
```
```
model = load_checkpoint('checkpoint.pth')
print(model)
```

This saves all the info you need to reload your model and current state

# 20 - Loading Image Data
You won't usually be working with the artificial datasets we've been using (MNIST)

```
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms

import helper
```

Easiest way to load image data is with `datasets.ImageFolder` from torchvision
  - `dataset = datasets.ImageFolder('path/to/data', transform=transform)`
  - expects format like `root/dog/xxx.png`, `root/dog/xxy.png`, `root/cat/123.png`
    - every class has it's own directory
    - images will be labeled with the class taken from the directory name

You'll need to transform the images to be the same size
  - `transforms.Resize()` to resize all the photos
  - `transforms.CenterCrop(), transforms.RandomResizedCrop()` for cropping the photos

Convert the images to PyTorch tensors with `transofrms.ToTensor()`

Usually best to do this in a pipeline as follows:
```
transform = transforms.Compose([transforms.Resize(255),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor()])
```

Data Loaders
  - after loading ImageFolder you must pass it to a DataLoader
  - takes the dataset and returns batches of images and corresponding labels
  - you can set batch size, whether to shuffle after each epoch, etc

`dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)`

The dataloader is a generator.
  - to get data out of it you need to loop through it or convert it to an interator and call `next()`

```
# Looping through it, get a batch on each loop 
for images, labels in dataloader:
    pass

# Get one batch
images, labels = next(iter(dataloader))
```

# 21 - Loading Image Data: Solution
Get all the imports in place:
```
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms

import helper
```

Load, transform and prep the data:
```
data_dir = 'Cat_Dog_data/train'

transform = transforms.Compose([transforms.Resize(255),
                               transforms.CenterCrop(224),
                               transforms.ToTensor()])

dataset = datasets.ImageFolder(data_dir, transform=transform)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
```

Data Augmentation
  - common strategy for training neural networks by introducing randomness in the input data itself
    - randomly rotate, mirror, scale, crop images during training

```
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5], 
                                                            [0.5, 0.5, 0.5])])
```

Use `transform.Normalize` to normalize your images
  - pass in list of means and list of standard deviations
  - the color channels are then normalized `input[channel] = (input[channel] - mean[channel]) / std[channel]`
  - subtracting mean centers data around zero, dividing by std squishes values between -1 and 1
  - NOTE: This makes backpropagation more stable (without it networks will often fail to learn)
  - for testing you want images that aren't altered (except for normalizing in the same way) so only resize/crop validation images

# 22 - Pre-Notebook with GPU
This shows us how to accelerate network computations using a GPU
  - our workspace will be GPU-enabled
  - our GPU hours are limited

The following is recommended:
  - work in CPU mode while developing your models
  - ensure the network is learning (training loss is dropping) using just CPU
  - when ready to train for real and optimize hyperparameters, enable GPU

NOTE: All models and data they see as input will have to be moved to the GPU device so note relevant movement code in model creation and training process

# 23 - Notebook Workspace with GPU
This is just a workspace so nothing to take note of

# 24 - Transfer Learning
We'll learn how to use pre-trained networks to solve challenging computer vision problems
  - in this case, networks trained on ImageNet (available from Torchvision)

Using a pre-trained netwok on images not in the training set is called transfer learning.

You can get the pretrained models by importing models from torchvision
  - most pretrained models require input to be 224x224
  - need to match the normalization from when the models were trained
    - means: `[0.485, 0.456, 0.406]`
    - std dev: `[0.229, 0.224, 0.225]`

# 25 - Transfer Learning Solution
Load in all the vars:
```
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
```

Set up your transforms:
```
data_dir = 'Cat_Dog_data'

# TODO: Define transforms for the training data and testing data
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

model = models.densenet121(pretrained=True)
```

```
# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, 500)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(500, 2)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier
```

This is now a very deep neural netwok so running it on a regular computer would take a long time.
  - we can use GPU to process at 100x faster

PyTorch (and most other frameworks) use CUDA to compute forward/backward passes on GPU
  - move you model paras and tensors to GPU memory using `model.to('cuda')`
  - move them back using `model.to('cpu')`

```
for device in ['cpu', 'cuda']:

    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    model.to(device)

    for ii, (inputs, labels) in enumerate(trainloader):

        # Move input and label tensors to the GPU
        inputs, labels = inputs.to(device), labels.to(device)

        start = time.time()

        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if ii==3:
            break
        
    print(f"Device = {device}; Time per batch: {(time.time() - start)/3:.3f} seconds")
```

Device = cpu; Time per batch: 5.678 seconds
Device = cuda; Time per batch: 0.010 seconds

Full Solution:
```
# Use GPU if it's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.densenet121(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False
    
model.classifier = nn.Sequential(nn.Linear(1024, 256),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(256, 2),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

model.to(device);

epochs = 1
steps = 0
running_loss = 0
print_every = 5
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
            running_loss = 0
            model.train()
```

# 26 - Tips, Tricks and Other Notes
Watch shapes
  - make use of `.shape` method during debugging and development to make sure tensor shape is what you expect/need

Network isn't training appropriately?
  - check that you're clearing gradients in training loop with `optimizer.zero_grad()`
  - if doing validation loop, make sure the network is set to evaluation mode with `model.eval()` and back to training mode with `model.train()`

CUDA errors
  - if you see:
  ```
  RuntimeError: Expected object of type torch.FloatTensor but found type torch.cuda.FloatTensor for argument #1 ‘mat1’
  ```
  notice that the second one has `torch.cuda`
    - means it's a tensor tht was moved to GPU but it's expecting tensor on the CPU
    - PyTorch needs tensors to be on the same device so make sure all things have been moved to appropriate place with `.to(device)`
