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



# 25 - Logistic Regression Algorithm



# 26 - Pre-Lab: Gradient Descent



# 27 - Notebook: Gradient Descent



# 28 - Perceptron vs Gradient Descent



# 29 - Continuous Perceptrons



# 30 - Non-Linear Data



# 31 - Non-Linear Models



# 32 - Neural Network Architecture



# 33 - Feedforward



# 34 - Backpropagation



# 35 - Pre-Lab: Analyzing Student Data



# 36 - Notebook: Analyzing Student Data



# 37 - Outro


