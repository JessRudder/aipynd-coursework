# 1 - Instructor
It's Luis again!

# 2 - Training Optimization
Things that can fail
  - poorly chosen architecture
  - noisy data
  - model could take forever to run

Let's learn how to optimize the training of our model

# 3 - Testing
Have training data and testing data
  - training data is labeled so the model can use it to adjust
  - testing data ensures that the training data hasn't overfit the model

Whenever there is a simple model that does almost as well as a more complicated model, prefer the simpler model

# 4 - Overfitting and Underfitting
Underfitting
  - trying to kill Godzilla with a flyswatter
  - approach is too simple to do the job
  - sometimes referred to as "error due to bias"
Overfitting
  - killing fly with bazooka
  - overly complicated and will lead to bad solutions
  - can fit the data well, but will fail to generalize
  - error due to variance

It's hard to get the architecture right
  - err on the side of a slightly more complicated model
  - then apply techniques to prevent overfitting

# 5 - Early Stopping
As you run through more epochs, the model gets more complicated and more fit to the training data
  - compare the training error with the testing error
    - if both are BIG, keep going
    - if both are small, stop
    - if training error is tiny but testing error is big, you done overfit son

If you plot the difference in training error with testing error, you get the "Model Complexity Graph"
  - y is error
  - x is number of epochs
  - you'll have curves for training data and testing data
  - where they start to diverge is where you want to stop training the model

Do gradient descent until the testing error stops decreasing and starts to increase
  - at that point, stop
  - this is called the Early Stopping algorithm and is used widely to train neural networks

# 6 - Regularization
Quiz about which basic model (separating two points) give the smaller error
  - I guessed but I was right so I win

# 7 - Regularization 2
x1 + x2 = 0 OR 10x1 + 10x2

Both give the same line (one is a scalar of the other)

sig(1+1)=0.88
sig(-1 - 1) = 0.12

sig(10 + 10) = 0.9999999979
sig(-10 - 10) = 0.0000000021

Second model gives the better error, but, it's overfitting in a subtle way
  - the simple line gives a better slope on the sigmoid function
  - the 10x line has a very steep slope between 0 and 1

"The whole problem with artificial intelligence is that bad models are so certain of themselves, and good models so full of doubts." - BertrAIND Russell

Large coefficients -> Overfitting
  - penalize large weights
  - take old error function and add a term which is big when the weights are big (two options)
    - L1 Regularization
      - add sums of absolute values of the weights times a constant lambda
      - Good for feature selection
      - end up with sparse vectors (small weights will tend to go to 0)
    - L2 Regularization
      - add the sum of teh squares of the weights times a constant lambda
      - Normally better for training models
      - maintains all the weights homogeniousl small

# 8 - Dropout
One part of the network could end up with large weights and dominate the training
  - you can turn the heavily weighted part of the network off and let the other parts train
  - as we go through some of the epochs, we randomly turn off nodes and don't allow data to pass through there
    - other nodes have to pick up slack and take more part in the training

Give the alogorithm a parameter
  - probability that each node will be dropped at a particular epoch
  * this is common and useful for training neural networks

# 9 - Local Minima
In a more complicated curve, gradient descent on it's own can get you stuck in a local minima
              _____
\   ___      /     \_/
 \_/   \    /
        \__/

Don't get stuck in one of the little cuppies when you want to be in the deepest one

# 10 - Random Restart
Start from different random places and do gradient descent from each of them
  - increases the probability that we'll get to the global minimum (or at least a good local minimum)

# 11 - Vanishing Gradient
Sigmoid function curve gets pretty flat at the sides
  - if you calculate the derivative at a point way to the right/left it's almost zero
  - problematic because derivative is what tells us what direction to move
  - with gradient descent alone we end up taking tiny, tiny steps that mean we'll never get down the mountain

# 12 - Other Activation Functions
Best way to fix vanishing gradient is to change the activation function

Hyperolic Tangent Function
tanh(x) = e^x - e^-x
          ----------
          e^x + e^-x

Simialr to sigmoid function, but, since the gradients are between 1 and -1, the slopes are more sloped
  - this lead to great advances in neural networks

Rectified Linear Unit (ReLU)
relu(x) = {x if x >= 0
          {0 if x <  0

If you're positive, I'll return the same number
If you're negative, I'll return 0
  - Gives you the maximum between x and 0

Can improve training significantly without sacrificing much accuracy since derivative is 1 if number is positive

With better activation funcitons, product will be made of slightly larger numbers which will allow us to do gradient descent

If you use one of these other models, you'll usually want the last node to be a sigmoid function since we'll need that to be a probability between 0 and 1

# 13 -  Batch vs Stochastic Gradient Descent
With lots of data, the matrix calculations can be huge and require a lot of memory and computing power

Do we need to plug in all our ddata every time we take a step?
  - if data is well distributed, a small subset would give us a good idea of what the gradient would be

Do the following:
  - Split data into several batches
  - Take points in first batch and run them through neural network
    - calculate error and gradient
    - backpropagate to update weights
  - Take points in next batch and do the same thing
  - Lather rinse repeat

Each of these steps will be less accurate than once through with all of the data
  - better to take many slightly inaccurate steps than one good one

# 14 - Learning Rate Decay
If learning rate is too big, you'll go far at first but you may miss the minimum and you model will get chaotic

With small learning rate, you'll make tiny steps and have a better chance of arriving to your local minimum
  - might make your model slow

General Rule: If your model is not working, decrease your learning rate

Best models usually decrease learning rate as you get closer to a solution
  - Keras has options to do this

# 15 - Momentum
Walk fast with determination
  - if you end up in a local minimum, you can power through and hopefully end up in a better solution

Do this by taking the average of the last 3-4 steps to determine the next step
  - weight each step so previous step matters a lot and the steps before matter less and less

Momentum
  - constant (beta) between 0 and 1
  - attaches to the previous steps as follows:
    - previous step * 1
    - one before * beta
    - one before * beta^2
    - one before * beta^3
    - etc
  - this should get you over local minimum humps but not push you out of the true global minimum

NOTE: Algorithms that use momentum seem to work really well in practice

# 16 - Error Functions Around the World
There are many other error functions that we didn't cover
  - it was just funny mountain names
  - not sure if there are related error functions that we should look up or if this was just silliness

