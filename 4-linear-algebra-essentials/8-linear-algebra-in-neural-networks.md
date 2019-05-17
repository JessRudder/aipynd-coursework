# 1 - Instructor
It's Ortal again!

# 2 - Brief Introduction
Linear algebra is very important for neural networks
  - keep going to find out more!

# 3 - What is a Neural Network?
In compsci we're referring to artificial neural networks
  - inspired by biological ones

Each network has
  - input neurons (referred to as input layer)
  - output neurons (referred to as output layer)
  - internal neurons (referred to as hidden layer)
    * each network can have many hidden layers

There is no connection between the number of inputs, number of hidden neurons or number of outputs

# 4 - How are the Neurons Connected?
The lines connecting the neurons symbolize a coefficient (scalar)
  - these are called weights
Lines connect each enuron in a specific layer to all the neurons in the following
  - since there are so many weights connecting one layer to the next, we organize those in a matrix called the "Weight Matrix"

Note: When training an ANN, we're looking for the best set of weights that will give us a desired outcome (more on this later!)

# 5 - Putting the Pieces Together
Each input is multiplied by its corresponding weight and added at the next layers neuron with a bias
  - bias is an external parameter of the neuron
  - can be modeled by adding an external fixed value input
Will then go through activation function to next layer/output
  - We'll learn about activation functions later

The Goal:
System is black box with n inputs and k outputs

Design the system in a way that it will give correct output y for a specific input x

We decide what's in the black box
  - use an ann and train it

We want to find the optimal set of weights connecting the input to the hidden layer and the hidden layer to the output

2 phases for ANNs:
Training
  - training set used to find weights that best maps to desired output
  - two steps
    - feedforward
    - backpropagation
Evaluation
  - use network created in training hase to apply new inputs and obtain desired outputs

# 6 - The Feedforward Process- Finding h
Use linear algebra to go from input to hidden state:
  - if you have more than one neuron in the hidden case, h is a vector

x1
x2    h1
x3    h2   (imagine all connected)
x4    h3   (by lines - weights)

x = |x1  x2  x3  x4|
w = |w11 w12 w13|
    |w21 w22 w23|
    |w31 w32 w33|
    |w41 w42 w43|

each vector x = vector x multipled by the weight matrix

Multiply input vector by weight matrix for a simple linear combination to calculate each neuron in the hidden layer

h'1 = x1*w11 + x2*w21 + x3*w31 + x4*w41
h'2 = x1*w12 + x2*w22 + x3*w32 + x4*w42
h'3 = x1*w13 + x2*w23 + x3*w33 + x4*w43
h'4 = x1*w14 + x2*w24 + x3*w34 + x4*w44

gives us h prime but then we need to use the activation function to find the actual hidden vector (Note: This wasn't covered in depth, just mentioned in passing)

Symbol we use for the activation function is the Greek letter phi: Φ
h = Φ(h')


# 7 - The Feedforward Process- Finding y
We found h and now we need to find y

h1
h2  y1
h3  y2

h = |h1  h2  h3|
w = |w11 w12|
    |w21 w22|
    |w31 w32|
Output will be y = |y1 y2|

y = h * w

To do a good job of calcualating y there should be many hidden layers (10+)

Notice that process for finding y is similar to process of finding h
  - each new layer in neural network is calcualted by matrix multiplication
    - vector == inputs to new layer
    - matrix == the connection to next layer
  
