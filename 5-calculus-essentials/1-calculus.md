# 1 - Our Goal
Learn calculus and figure out how it's related to machine learning.

# 2 - Instructor
Grant Sanderson
  - created 3Blue1Brown
  - former content creator at Khan Academy

# 3 - Introduction Video
Goal is for us to feel like we could have invented calculus

Integrals and Derivatives are opposites.

Walk through visually deriving area of a circle (2pi*r)
  - Start with hard problem => turn it into the sum of many values => Area under graph
  - same can be done with many other hard problems in science and math

Integral
  - formula that gives you the area under a graph at some point

Derivative
  - measure of how sensitive a function is to small changes in its input
  - key to solving integral questions (finding the area under a curve)

# 4 - Derivatives
Derivative
  - measure of an instantaneous rate of change (but this is an oxymoron)
  - looked at approximations for velocity (measuring change over tiny amount of time)
  - real derivative of velocity is the value as dt (amount of time) gets smaller and smaller (approaches 0)
  * pure math derivative is equal to slope of line tangent to graph at a single point

# 5 - Derivatives through Geometry
Tiny nudges are at the heart of derivatives

dx is so small that you can ignore anything that's dx raised to a power greater than 1

derivatives follow regular pattern for polynomials

d(x^1) = 1x^0
d(x^2) = 2x^1
d(x^3) = 3x^2
d(x^4) = 4x^3
d(x^n) = nx^n-1 (POWER RULE!)

d(sin(theta)) = cos(theta)

# 6 - The Chain Rule
The derivative of a sum of two functions is the sum of their derivatives

The derivative of the product of two functions
  - Left * d(Right) + right * d(Left)
  - left equation * derivative of right + right equation * derivative of left
  d/dx * (sin(x)x^2)
  sin(x)2x + x^2*cos(x)

Function composition (placing a function inside another one)
  - e.g. g(x) = sin(x)  h(x) = x^2
         g(h(x)) = sin(x^2)
                 = d(sin(x^2)*d(x^2)
                 = cos(x^2)*2x
  - derivative of outide function times derivative of inside function

# 7 - Derivatives of Expontentials
2^(t+dt) = 2^t*2^dt

d(2^t) = 2^t * some constant (0.6931372)
  - the constant is called the "proportionality constant"
    * NOTE: This constant is the natural log of the base (in this case ln(2))
  - this pattern holds for other exponents
    - d(3^t) = 3^t * constant
    - the constant changes for each equation
  - e is the value whose proportionality constant is 1
    - d(e^t) = e^t

Note: In calculus you will rarely see exponentials writen as base ^ t instead you'll always see e^(const * t)

# 8 - Implicit Differentiation
Implicit Curve
  - set of all points x,y that satisfies some property written in terms of the two variables x and y

Implicit Differentation
x^2 + y^2 = 5^2
2xdx + 2ydy = 0
dy/dx = -x/y

x(t)^2 + y(t)^2 = 5^2
d(x(t)^2 = y(t)^2)/dt
2x(t)*(dx/dt) + 2y(t)*(dy/dt) = 0

sin(x)y^2 = x
sin(x)*(2ydy)+y^2*(cos(x)dx) = dx
  - if the step is going to keep us on the curve, the right and left must change by the same
  - next step is to commonly solve for dy/dx

y = ln(x)
e^y = x
e^ydy = dx
dy/dx = 1/e^y
e^y is same as x so....
slope = dy/dx = d(ln(x))/dx = 1/x

This is a sneak peek at multivariable calculus!

# 9 - Limits
Assigning fancy notation to intuitive idea that one number gets closer to another.

Epsilon, Delta definition of limits
  - not really important for now, but is fun to know for later when we're all into the calculus

Talked about L'hopital's rule
  - I definitely did not understand this

# 10 - Integrals
They are an inverse of derivatives

S8/0 v(t)dt
  - S8/0 is the funky large s with 8 at top and 0 at bottom
v(t) = t(8-t)
described the area under the graph
== Integral (because it integrates the area under the graph)

Derivative of any funtion giving the area under a graph is equal to the function for the graph itself
Derivative: 8t-t^2
Antiderivative: 4t^2-1/3*t^3 + C
  - infinite number of Cs that would move graph up and down but not change the shape
  - subtractthe off the value of that antiderivative function ta the lower bound
    - in this case, plug in zero which = zero so subtract a constant of 0
    - won't always be 0 though!

Any time you want to integrate some function 
  - first step to evaluating that integral is to find an antiderivative (some other function whose derivative is the thing inside the integral)
  - integral = antiderivative evaluated at the top bound minus its value at the bottom bound
  * This is the Fundamental theorem of calculus

# 11 - More on Integrals
Let's learn how to find the average of a continuous variable
  - e.g. the average height of the graph in a given interval
  - vague sense that we want to add up an infinite set of numbers and divide by infinity
  - that sense almost always means we should use an integral

# 12 - The Taylor Series (optional)
Taylor Series
  - representation of a function as an infinite sum of terms
  - a way to proximate a function
  * Not needed to build/train our own neural network

# 13 - Multivariable Chain Rule
What happens when there's more than one variable in our function?
  - In neural networks this will almost always be the case
Comes up any time you have one variable influencing another variable along multiple different paths
  - x determines values of f and g which then combine to determine value of h
      f = x^2
    /   \
  x       h f^2g
    \   /
      g = cos(pi*x)
  - assume we need to understand how a small change to x will affect h

To solve:
  - calculate all of the derivatives associated with the edges
  - dh/dx = df/d * Gh/Gf + dg/dx * Gh/Gg
    * NOTE: G means partial derivative because the actual derivative requires more than one input
  - (2x)(2fg)*(-sin(pi*x)*pi)(f^2)
  - 32 (meaning tiny nudge to x results in a 32 times larger change to h)
