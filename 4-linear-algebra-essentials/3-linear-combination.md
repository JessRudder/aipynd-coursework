# 1 - Linear Combination: Part 1
Two special vectors in linear algebra
i^ (i-hat) = |1|
             |0|

j^ (j-hat) = |0|
             |1|

2d vectors can be described as sum of 2 scaled vectors

x = |3|
    |2|

x = 3*i^ + 2*j^

i^ and j^ are the `basis vectors`

You could have chosen different `basis vectors` and ended up with a difference coordinate system (but we chose i^ and j^)

Any time you're scaling 2 vectors and adding them, it's a `linear combination` of those 2 vectors.

The set of all possible vectors you can reach with a linear combination of a given pair of vectors is called the span of those two vectors
  - span of most 2d vectors is all 2d space
  - if vectors line up, span is just a single line


# 2 - Linear Combination: Part 2
If you're thinking about a vector on it's own, think of it as an arrow

If thinking of a set of vectors, think of them all as points
  - e.g. span of most 2d vectors ends up being entire infinite sheet of 2d space (all points on a plane)

Span of 3d vectors
  - think of 2d span as flat sheet cutting through 3d space (as long as none of the vectors point in the same direction)
  - if 3rd vector is on span of first 2, span doesn't change
  - if 3rd vector is not on span of first 2, that sheet moves around covering all of 3d space

If you have multiple vectors lined up (where on is redundant and doesn't change the span), you say that they are `Linearly dependent`

If each vector adds value to the span, they are `Linearly independent`


# 3 - Linear Combination and Span
linear combination is multiplicationo f a scalar to a variable and addition of those terms

x, y, z are variables
a1, a2, a3 are scalars
following equation will be linear combination

v = a1x + a2y + a3z

Now assume x, y, z are vectors and a1, a2, a3 are scalars and the equation is the same

The span of vectors v and w is the set of all of their linear combinations

# 4 - Linear Combination: Quiz 1
Checked our understanding of when the span of vectors is a plane vs when it's a line

# 5 - Linear Dependency
When one vector can be defined as a linear combination of the other vectors, they are a set of linearly dependent vectors.

When each vector in a set of vectors vector can not be defined as a linear combination of the other vectors, they are a set of linearly independent vectors.

Note: Easiest way to know if set of vectors is linearly dependent is with use of determinants (out of scope of this course)

# 6 -  Solving a Simplified Set of Equations

Assume 2 vectors:
x = |-14|  y = | 5|
    |  2|      |-1|

We want to represent a new vector as a linear combination of x and y

New Vector = |-13|
             |  3|

a|-14| + b| 5| = |-13|
 |  2|    |-1|   |  3|

|-14a| + | 5b| = |-13|
|  2a|   |-1b|   |  3|

End up with 2 equations (set of two equations with 2 unknowns):

-14a + 5b = -13 and 2a - b = 3

Can be solved in a couple of ways:
1) Graphical solution - draw both lines and find the intersection

2) Substitution - Isolate a variable from one of the equations and substitute it in the second

-14a + 5b = -13
2a - b = 3 => b = 2a - 3

-14a + 5(2a - 3) = -13
a = -0.5

substitute a into the equation above to solve for b

b = -4

3) Elimination - Eliminate one of the variables by enforcing the same absolute value to one of the scalars/coefficients

-14a + 5b = -13
2a - b = 3

Multiply equation 2 by 5 to get

10a - 5b = 3

Now ad the two equations to get:

-4a + 0b = 2 => -4a = 2 => a = -0.5

# 7 - Linear Combination: Quiz 2
When equations are linearly dependent, there are an infinite number of solutions to the linear combination question

# 8 - Linear Combination: Quiz 
When vectors are parallel, there is no vector you represent a new vector as a linear combination of them
