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


# 15 - Fashion-MNIST: Solution


# 16 - Inference and Validation


# 17 - Validation: Solution


# 18 - Dropout: Solution


# 19 - Saving and Loading Models


# 20 - Loading Image Data


# 21 - Loading Image Data: Solution


# 22 - Pre-Notebook with GPU


# 23 - Notebook Workspace with GPU


# 24 - Transfer Learning


# 25 - Transfer Learning Solution


# 26 - Tips, Tricks and Other Notes

