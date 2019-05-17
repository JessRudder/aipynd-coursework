# 1 - What is a Matrix?
Two dimensional array that contains same elements as vector
  - can have m rows and n columns
  - is called m x n matrix

# 2 - Matrix Addition
In order to add matrices, the following must be true:
  - matrices must have same dimensions
  - add elements in correct corresponding indices

x = |a11 a12 a13|  y = |b11 b12 b13|
    |a21 a22 a23|      |b21 b22 b23|
    |a31 a32 a33|      |b31 b32 b33|

x+y = |a11+b11 a12+b12 a13+b13|
      |a21+b21 a22+b22 a23+b23|
      |a31+b31 a32+b32 a33+b33|

# 3 - Matrix Addition Quiz
Tested my knowledge of adding and subtracting matrices.

# 4 - Scalar Multiplication of Matrix and Quiz
To multiply by a scalar:
  - do not need to verify dimensions or indeces
  * multiply each element by the scalar

x = |a11 a12 a13|
    |a21 a22 a23|
    |a31 a32 a33|

8x = |8xa11 8xa12 8xa13|
     |8xa21 8xa22 8xa23|
     |8xa31 8xa32 8xa33|

# 5 - Multiplication of a Square Matrices
Must consider dimensions of each as multiplication is not possible if dimensions are not aligned

Easiest is 2 square matrices of the same dimensions

p = |p11 p12 p13| q = |q11 q12 q13|
    |p21 p22 p23|     |q21 q22 q23|
    |p31 p32 p33|     |q31 q32 q33|

p*q = |p11q11 + p12q21 + p13q31   p11q12 + p12q22 + p13q32   p11q13 + p12q23 + p13q33|
      |p21q11 + p22q21 + p23q31   p21q12 + p22q22 + p23q32   p21q13 + p22q23 + p23q33|
      |p31q11 + p32q21 + p33q31   p31q12 + p32q22 + p33q32   p31q13 + p32q23 + p33q33|

Each element in PxQ is result of multiplying all elements in a row i (from P) with the corresponding j elements in column j (from Q)

When two leanear transformations are represented by matrices, the matrix product represents composition of the two transformations
 

# 6 - Square Matrix Multiplication Quiz
Tested ability to multiply perfectly square matrices

The order things are written in matters:
AxB may not equal BxA (first is one that will provide the row and second is one that will provide the column)

# 7 - Matrix Multiplication: General
Can multiply matrices with different shapes as long as the number of rows in the first matches up with the number of columns in the second

p = |p11 p12 p13| q = |q11 q12 q13 q14|
    |p21 p22 p23|     |q21 q22 q23 q24|
                      |q31 q32 q33 q34|

p*q = 
|p11q11+p12q21+p13q31   p11q12+p12q22+p13q32   p11q13+p12q23+p13q33   p11q14+p12q24+p13q34|
|p21q11+p22q21+p23q31   p21q12+p22q22+p23q32   p21q13+p22q23+p23q33   p21q14+p22q24+p23q34|

# 8 - Matrix Multiplication Quiz
A matrix with only one row or column is a vector!
  - 1 row = row vector
  - 1 column = column vector

They threw in a trick question here. I got it, but, I also thought I might have been confused. Essentially, a 5X3 matrix cant be multipled by a 1X5 because you need the elements in the row on the first (3) to match up with the columns on the second (1).

You can go the other direction though 1X5 times 5X3 because you need the elements in the row on the first (5) to match up with the columns on the second (also 5)

# 9 - Linear Transformation and Matrices: Part 1
Linear Transformation
  - transformation is a function (takes in inputs and spits out output for each one)
  - in linear algebra, think of function that takes in some vector and spits out another vector
  * transformation suggests that you think using movement
  - linear if it has two properties
    - all lines must remain lines (no curves)
    - origin must remain fixed in place (keeps lines parallel and evenly spaced)

# 10 - Linear Transformation and Matrices: Part 2
To represent it numerically, you only need to know where the two basis vectors land (i-hat and j-hat)

All vectors start out as a linear combination of i-hat and j-hat and, post transformation, end up as that same combination of the now transformed i-hat and j-hat

vector |x| => x|i11| + y|j11|
       |y|     |i21|    |j21|

2d linear transformation is completely described by just 4 numbers
  - 2 coords for where i-hat lands
  - 2 coords for where j-hat lands

Commonly get packaged together into a single 2x2 matrix

|i11 j11|
|i21 j21|

# 11 - Linear Transformation and Matrices: Part 3
|a b| i-hat = |a|  j-hat = |b|
|c d|         |c|          |d|

Apply to vector |x|
                |y|

x|a| + y|b| = |ax+by|
 |c|    |d|   |cx+dy|

Shear - one basis vector stays fixed while the other moves

Note: If a transformation shifts the basis vectors until they are linearly dependent, all of 2d space will be squished into a line

Every time you see a matrix, it can be interpreted as a certain transformation of space

# 12 - Linear Transformation Quiz Answers
Nothing too exciting here. But, always good to have the answers.
