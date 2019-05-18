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



# 10 - Integrals



# 11 - More on Integrals



# 12 - The Taylor Series (optional)



# 13 - Multivariable Chain Rule


