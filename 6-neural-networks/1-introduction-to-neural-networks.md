# 1 - Instructor
Luis Serrano
  - machine learning engineer at Google
  - PhD in mathematics from U Mich
  - Postdoc at U Quebec

# 2 - Introduction
What is Deep Learning?
  - at the heart it's neural networks
  - given some data in the form of blue and red points, the neural network will look for the best line that separates them
What is it used for?
  - all the things


# 3 - Classification Problems 1
Look at a graph of points and identify where the point in question falls on the graph
  - student would get accepted

# 4 - Classification Problems 2
The line between the blue and red lines is the 'model'
  - if you are above the line you get accepted
  - below the line you get rejected
  - How do you find the line?

# 5 - Linear Boundaries
Find the equation for the 'model' and you can then use that to calculate if an individual would fall above or below it
  - positive number means you're in, negative means you're out

# 6 - Higher Dimensions
n-dimensional space
  - vector x1....xn
Boundary is n-1 dimensional hyperplane
  - equation is w1x1 + w2x2 + ... + wnxn + b = 0
  - can be abbreviated to Wx + b = 0
Prediction is same as before:
y-hat = 1 if Wx + b >= 0
y-hat = 0 if Wx +b < 0

# 7 - Perceptrons
Perceptron
  - buiding block of neural networks
  - encoding of our model equation into a small graph
Build it as follows
  NOTE: Our equation is 2*test + 1*grades - 18
  - fit data and boundary line into a node
  - add small nodes for the inputs (test/grades)
  - perceptron checks if point 7, 6 is in the positive or negative area
    - returns yes if positive
    - returns no if negative
  - the 2 and 1 are weights so we add those to the lines connecting nodes
  - the -18 is added to the large node

  test = 7
            \2
              (-18) = Accepted
            /1
  grade = 6 

This turns into:
7*2 + 6*1 - 18 = 2

The perceptron evaluates on a step function:

y = 1 if x >= 0
y = 0 if x < 0


# 8 - Why "Neural Networks"?
Perceptrons kind of look like neurons in the brain

# 9 - Perceptrons as Logical Operators
Some logical operators can be represented as perceptrons

Inputs can be true or false
Output can only be true if both inputs are true

Input 1
        \1  -2 (bias) 
           AND
        /1
Input 2

Inputs can be true or false
Output will only be false if neither input is true

Input 1
        \1  -1 (bias)
           OR
        /1
Input 2


Output will be the opposite of the input on only a single branch (in this case, the 2nd 1)

Input 1
        \0  1 (bias)
           NOT
        /-1.1
Input 2


XOR requires multilayer perceptron but can still be done.


# 10 - Perceptron Trick
In reality, we need to give the ML model the result and it will build the perceptron itself

Start by drawing an arbitrary line

Move the line depending on if it's doing well or not
  - if you're a misclassified point, you want the line to come closer to you

Lather, rinse, repeat

Math for moving the line:

Assume original line of
2(x1) + 4(x2) - 10 = 0

and a misclassified point of (4,5)

That gives us 
3     4     -10
4     5       1 (1 is bias)
------------------
-1    -1     -11

This would move our line dramatically in the direction of the misclassified line
  - good for that line, but possibly bad for the other points

Use a learning rate to make smaller moves toward the misclassified point
  - small number like 0.1

That gives us 
3      4      -10
4(.1)  5(.1)  1(.1)
------------------
2.6    3.5   -10.1

New line:
2.6(x1) + 3.5(x2) - 10.1 = 0

For a point that's misclassified below the line, you add instead of subtract

Do this over and over again until the point is in the proper area.

# 11 - Perceptron Algorithm
Start with random weights (w1...wn) and b
  - gives line Wx + b (and all the points are classified)

For every misclassified point (x1...xn):
  If prediction = 0
    - For i - 1...n
      - change wi to wi + axi (a = learning rate)
      - change b to b + a
  If prediction = 1
    - For i - 1...n
      - change wi to wi -  axi (a = learning rate)
      - change b to b - a

Repeat until you have either no errors or a number of errors that's small enough we're ok with it
  - could also say, "Do this 1000x then stop"

# 12 - Non-Linear Regions
Often the accept/reject region will not be made with a simple line (but will use a curve!)
  - need to refine our perceptron algorithm so it can generalize to curves

# 13 - Error Functions
We'll now solve our problems with the help of an error function
  - something that tells us how far we are from the solution
  - e.g. trying to get to a plant, the error function will tell you how far away from the plant you are


# 14 - Log-loss Error Function
Gradient descent
  NOTE: In order to do gradient descent, our error function must be continuous (like height) and not discrete (like number of incorrectly categorized nodes)
  Imagine being on top of cloud covered mountain
    - you want to descend but you cant see best way
    - consider all ways around you
    - pick direction that makes you descend the most
    - move that direction
    - repeat the process
    * in this case, the height is the error (decrease the error and you move closer to where you want to be)
  It's possible to be stuck in valley (local minimum)
    - there are ways of addressing this that we'll learn later
    - NOTE: Local minimum often gives good solution to problem
  Back to class situation
    - drawn random line to split
    - number of incorrectly classified students becomes the error
      - move line to decrease that 'error'
      - could end up moving small amounts for forever because everywhere you move the error is still two (unless you luck into slowly moving it past a node)
    - Make the incorrectly classified nodes into a continuous error by assigning a large weight to the wrong nodes and a small weight to the correct nodes then add all those weights together as the error
      - now, as you move the line around, the error changes based on how close you are to being correctly classified

# 15 - Discrete vs Continuous
We need a discrete error function in order to use gradient descent
  - must move from discrete predictions to continous predictions
    - discrete is yes/no (either the node is correctly classified or not)
    - continuous is value between 0 and 1 (probability that your right)
  - farther a point is from the black line the more likely it is to be classified correctly
    - if it's far into the yes section, it will be nearly 1
    - if it's far into the no section, it will be nearly 0
    - points around the line will be close to .5

s(Wx + b) = prediction for that node (0..1)

# 16 - Softmax
Softmax Function is the equivalent of the sigmoid activation function but for when problem has 3 or more classes
  - which animal did we see, duck/beaver/walrus?
  - calculate linear function based on those inputs to get score for each of the animals
    - need to turn this into number between 0 and 1
    - e^num1 / (e^num1 + e^num2 + e^num3) to get average that accounts for negative numbers

Formal Definition
  - linear function of scores Z1....Zn

P(class i) = e^Zi
           ---------
          e^Z1+...+e^Zn

# 17 - One-Hot Encoding
Input data will not always look like numbers
Gift Probability
Yes 1
No  0

Animal Probability
        duck?   beaver?  walrus?
Beaver    0       1        0
Duck      1       0        0
Walrus    0       0        1

Adds more columns to the database, but, this is the common way to handle multiple variables

# 18 - Maximum Likelihood
Maximum Likelihood
  - pick the model that gives the existing labels the highest probability
  - e.g. if a student was labeled as getting in and the model said they would get in, it's better than a model that said they wouldn't get in

If the events are independent, the probability for the entire graph is p1 * ... * pn

# 19 - Maximizing Probabilities
Better models give us better probabilities
  - compare two probabilities and then go toward the one that gives you the best probability
  - problem: Probability is a product of numbers and products are hard
    - if you're doing the product of thousands of numbers between 0 and 1 it could end up very very tiny
Products are bads but sums are good
  - turn the products into sums with log!

# 20 - Cross-Entropy 1
log(a*b) = log(a) + log(b)

Take the products and take the log of each input and add them together

ln(p1) + ln(p2) + ln(pn)
  - ln of these numbers will be negative, so, we actually want:

-ln(p1) - ln(p2) - ln(pn)

A good model will give us a low cross-entropy

Can even look at the negative ln at each point as an error at each point
  - high probability will result in low ln (low error)
  - low probability will result in high ln (high error)

No longer trying to maximize a probability...Now we're trying to minimize a cross entropy

# 21 - Cross-Entropy 2
Cross Entropy
  - if I have a bunch of events and a bunch of probabilities, how likely is it that those events happened based on those probabilities
    - very likely == small cross entropy
    - very unlikely == large cross entropy

# 22 - Multi-Class Cross Entropy
What happens if we have more classes (not just a yes or no)

animal door1 door2 door3
duck     .7    .3    .1
beaver   .2    .4    .5
walrus   .1    .3    .4

Assume doors are as follows

  door1   door2    door3
  duck    walrus   walrus
P= .7  *    .3   *  .4 = 0.084
CE= -ln(.7) - ln(.3) - ln(.4) = 2.48

# 23 - Logistic Regression
Logist Regression Algorithm
  - Take data
  - Pick random model
  - Calculate the error
  - Minimize the error and obtain a better model
  - Enjoy!

Minimize the error function
  - jiggle the line around to minimize the error function
  - use gradient descent for this

# 24 - Gradient Descent
How to get down?
  - input of the functions are W1 and W2
  - error function is given by E
  - gradient of E is given by vector sum of the partial derivatives of E with respect to W1 and W2
  * This will tell us which direction to move if we want to increase the error function the most (by taking the negative of the gradient)

y-hat = sigma(Wx+b)
y-hat = sigma(w1x1 + ... + wnxn + b)
DeltaE = (pdE/pdw1...pdE/pdwn*pdE/pdb)
*pd is partial derivative of

Take the negative of that for the gradient descent
  - don't want to take a huge step, so, add a learning rate (alpha)
  - -alpha*deltaE

Update the weights
  - wi' <- wi - alpha * pdE/pdwi
  - b' <- b-alpha * pdE/pdb

Sigmoid has nice derivative
sigma'(x) = sigma(x)*(1-sigma(x))

If you need a walkthrough of the calculation go [here](https://classroom.udacity.com/nanodegrees/nd089/parts/52fefcaa-2550-4581-87cd-2347fa527447/modules/8d44653f-dfda-4720-88ba-cfa77a93c009/lessons/93d158ce-25e1-4fc1-a187-162982e3cef7/concepts/0d92455b-2fa0-4eb8-ae5d-07c7834b8a56)

Small gradient means we change our coordinate by a little bit

Large gradient means we change our coordinate by a lot

# 25 - Logistic Regression Algorithm
Pseudo code for gradient descent algorithm
  - start with random weights w1 up to 1n and b
    - gives us a line
    - is actually the whole probabiility function given by sigma(Wx+b)
  - calculate error for every point
    - large for misclassified points
    - small for correctly classified points
  - for every point (x1...xn)
    - Update Wi with wi - alpha * pdE/pdwi (new weights wi')
      - wi - alpha(y-hat - y)xi
    - update b with b - a * pdE/pdb (new bias b')
      - b - alpha(y-hat - y)
  - repeat until the error is small
    - number of times you repeat is called epochs

This is suspiciously similar to the Perceptron Algorithm

# 26 - Pre-Lab: Gradient Descent
Instructions for completing the lab

# 27 - Notebook: Gradient Descent
Recommend coming back [here](https://classroom.udacity.com/nanodegrees/nd089/parts/52fefcaa-2550-4581-87cd-2347fa527447/modules/8d44653f-dfda-4720-88ba-cfa77a93c009/lessons/93d158ce-25e1-4fc1-a187-162982e3cef7/concepts/64f025bd-1d7b-42fb-9f13-8559242c1ec9) and working through this if we have trouble implementing the classifier

# 28 - Perceptron vs Gradient Descent
The numbers for each work out to be very similar

Misclassified Point
  - Both say come closer!
Correctly Classified Point
  - Perceptron says do nothing
  - Gradient Descent says go farther away

# 29 - Continuous Perceptrons
Went back over perceptrons (with a 30,000 ft view)

# 30 - Non-Linear Data
Some data requires non linear boundaries in the model
  - we'll learn about that in the next video

# 31 - Non-Linear Models
We'll still use gradient descent

Everything will be the same as before

But, the equation will be a curve, not a line

# 32 - Neural Network Architecture
Neural Networks == Multi-Layer Perceptrons

To create one non-linear model, combine two linear models
  - this line + this line = that curve

Combining Models
  - Get both models
  - Calculate the probability for a given point in each model
  - Add the probability of the point in each model together (might be greater than one which desnt work for a probability)
  - apply sigmoid function to sum of probabilities to turn it into a number between 0 and 1

YOu can add weights to favor one model over another
  - e.g. 7*prob of point 1 + 5*prob of point 2 then run sum through a sigmoid function

You can add a bias too if you want

5x1 - 2x2 + 8
x1
   \5
      (-8)     \
  /-2           \7
x2               \
                  (-6)
7x1 - 3x2 - 1    /
x1              /5
   \7          /
      (1)     /
  /-3
x2

Now we have a whole neural network!

Neural Networks can get a lot more complicated
  - you can add more nodes to the input, hidden and output layers
  - add more layers

Neural networks
  - first layer is input (x1, x2, xn)
  - hidden layer (set of linear models created with first input layer)
  - output layer (linear models are combined to obtain nonlinear model)

They can get more complicated
  - use 3 hidden layer models to create a triangular output!
  - add more nodes (we're not in 2D space any more)
  - output could have more nodes (multiclass classification model)
  - add more hidden layers (this gives us a deep neural network)
    - linear models combine to create very complicated output models
    - gives us highly complex models

Binary Classification
  - two options
    - either a 1 or a 0
    - get the gift or not

Multi-Class Classification
  - multiple options
    - duck, beaver, walrus
  - could create one neural network for each option then choose the option with the highest predicted probability
  - can do it with a single neural network
    - first layers will tell us about the image
    - last layer will tell us what it is
  - there will be an output node for each option
    - it will tell us the probability that the image is that animal
    - take those probabilities and apply the SoftMax function

# 33 - Feedforward
Process neural networks use to turn input into an output

Training a Neural Network
  - what parameters should they have on edges in order to model our data well

Feedforward Process
  - take probabilites from both models, combine them, plot the point with the new model and calculate it's probability (both examples they show gave a bad model with a misclassified point)

First layer vector * weights and bias matrix = outputs
Apply sigmoid function to outputs to get a value between 0 and 1
Multiply that by second weights and bias matrix
Throw that output into a sigmoid function to get the final probability (y-hat)
  - the probability that the point is labeled blue

This combines to make a highly non-linear model

Error Function
  - same as before, our goal is to minimize it
  - it's a complicated formula that involves a lot of symbols...I'm guessing you'll have to look it up when you need to use it

# 34 - Backpropagation
Train your neural network with Backpropagation
  - do feedforward operation
  - compare output of model with the desired output
  - calculate the error
  - run the feedforward operation backward to spread error to each of the weights
  - use this to update the weights and get a better model
  - continue until we have a model that is good

Run through feedforward then ask a point, "what should the model do for you to be better classified?"
  - look at previous layer and see which one had the point classified better (bump up that model's weight)
  - go back another layer and ask the models what each of them could do to better classify the point
  - that will update the weights of the initial input layer

Lots of complicated mathy stuff.
  - supposedly Keras (Python's deep learning library) will take care of most of this for us

# 35 - Pre-Lab: Analyzing Student Data
Info about the upcoming lab
  - you'll do the following:
    - One-hot encoding the data
    - scaling the data
    - writing the backpropagation step

# 36 - Notebook: Analyzing Student Data
We can come back to this later if we have time

# 37 - Outro
Now we know all about neural networks

Next Up - how to do this with NumPy

