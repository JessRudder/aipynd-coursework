# 1 - Mean Squared Error Function
Log-Loss vs Mean Squared Error
  - previously learned log-loss function
  - many other error functions used for neural networks
  - mean squared error is another!

It's the mean of the squares of the differences between the predictions and the labels

# 2 - Gradient Descent
Gradient Descent with Squared Errors
  - we want to find the weights for our neural network
  - need the error to tell us when we're wrong

Common metric is sum of the squared errors (SSE)
  - sum the squared differences between actual y and predicted y
  - calculate that for each point and sum it up
  - this gives overall error for all output predictions for all the data points

SSE is good for a few reasons
  - square ensures error is always positive
  - larger errors penalize more than smaller errors
  - makes the math nice

We want the error to be as small as possible
  - the weights are the knobs we turn to adjust the error

Gradient Descent
  - gradient is derivative generalized to functions with more than one variable

At each step calculate the error and the gradient
  - use those to determine how much to change each weight
  - this will eventually help you find weights close to the minimum of the error function

Caveats
  - If weights are initialized with wrong values, you can end up in a local minimum instead of the actual minimum
  * NOTE: You can avoid this using methods such as [momentum](https://distill.pub/2017/momentum/)

# 3 - Gradient Descent: The Math
Video is walking us through the math we talked about a couple lessons ago

Sum of the Squared Errors
  - measure of network's performance (high == making bad predictions)

  E = 1/2 * sum (y - y-hat)^2
  (where y is actual and y-hat is prediction)

[This video](https://classroom.udacity.com/nanodegrees/nd089/parts/52fefcaa-2550-4581-87cd-2347fa527447/modules/8d44653f-dfda-4720-88ba-cfa77a93c009/lessons/0e07fafa-e796-4fab-b119-13f47f1d5c1b/concepts/3156ccf8-9bd0-4019-83b9-ab39c53bf541) is very helpful for understanding the math of gradient descent

# 4 - Gradient Descent: The Code

Code for calculating gradient descent when there is only one output unit

```
# Defining the sigmoid function for activations
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Input data
x = np.array([0.1, 0.3])
# Target
y = 0.2
# Input to output weights
weights = np.array([-0.8, 0.5])

# The learning rate, eta in the weight step equation
learnrate = 0.5

# the linear combination performed by the node (h in f(h) and f'(h))
h = x[0]*weights[0] + x[1]*weights[1]
# or h = np.dot(x, weights)

# The neural network output (y-hat)
nn_output = sigmoid(h)

# output error (y - y-hat)
error = y - nn_output

# output gradient (f'(h))
output_grad = sigmoid_prime(h)

# error term (lowercase delta)
error_term = error * output_grad

# Gradient descent step 
del_w = [ learnrate * error_term * x[0],
          learnrate * error_term * x[1]]
# or del_w = learnrate * error_term * x
```

# 5 - Implementing Gradient Descent
How do we translate that code to calculate many weight updates so our network will learn?

First step is to prepare the data
  - they will go over this with us later

Mean Square Error
  - take the mean of the squared errors (instead of the sum)
  - this ensures you don't take too large of a step when you have lots of data

General Algorithm for updating weights with gradient descent:
  - set weight step to zero
  - for each record in the training data
    -make a forward pass through network calculating y-hat
    - calculate error ter for output unit
    - update weight step
  - update weights (averaging weight steps to reudce large variations in training data
  - repeat for e epochs)

Implementing in NumPy
```
weights = np.random.normal(scale=1/n_features**.5, size=n_features)
# input to the output layer
output_in = np.dot(weights, inputs)
weights += ...
```

They wanted us to implement this, but, I don't have time!

# 6 - Multilayer Perceptrons
Going over multilayer again
  - output of one layer is used as the input of the layer that follows

Create your weight matrix in NumPy:

```
# Number of records and input units
n_records, n_inputs = features.shape
# Number of hidden units
n_hidden = 2
weights_input_to_hidden = np.random.normal(0, n_inputs**-0.5, size=(n_inputs, n_hidden))
```

Then you multiply the input vector by the weight matrix
```
hidden_inputs = np.dot(inputs, weights_input_to_hidden)
```

Asked us to implement a forward pass from input to output layer

# 7 - Backpropagation
Error for units is proportional to the error in the input layer times the weight between the units

Visualize back propagation by flipping the network over and running as usual (but now you're feeding backwards)

Asks us to code Backpropagation:

```
import numpy as np


def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


x = np.array([0.5, 0.1, -0.2])
target = 0.6
learnrate = 0.5

weights_input_hidden = np.array([[0.5, -0.6],
                                 [0.1, -0.2],
                                 [0.1, 0.7]])

weights_hidden_output = np.array([0.1, -0.3])

## Forward pass
hidden_layer_input = np.dot(x, weights_input_hidden)
hidden_layer_output = sigmoid(hidden_layer_input)

output_layer_in = np.dot(hidden_layer_output, weights_hidden_output)
output = sigmoid(output_layer_in)

## Backwards pass
## TODO: Calculate output error
error = target - output

# TODO: Calculate error term for output layer
output_error_term = error * output * (1 - output)

# TODO: Calculate error term for hidden layer
hidden_error_term = np.dot(output_error_term, weights_hidden_output) * \
                    hidden_layer_output * (1 - hidden_layer_output)

# TODO: Calculate change in weights for hidden layer to output layer
delta_w_h_o = learnrate * output_error_term * hidden_layer_output

# TODO: Calculate change in weights for input layer to hidden layer
delta_w_i_h = learnrate * hidden_error_term * x[:, None]

print('Change in weights for hidden layer to output layer:')
print(delta_w_h_o)
print('Change in weights for input layer to hidden layer:')
print(delta_w_i_h)
```

# 8 - Implementing Backpropagation
Another opportunity to code through an example problem

# 9 - Further Reading

