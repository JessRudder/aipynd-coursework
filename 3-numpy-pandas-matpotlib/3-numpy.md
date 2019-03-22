# 1 - Instructors
Juan Delgado
  - computational physiscist with MA in Astronomy
Juno Lee
  - teacher from before

# 2 - Introduction to NumPy
Numerical Python
  - fundamental package for scientific computing in Python
  - extensive math library
This Course:
  - uses numpy 1.13
  - check installed version with `conda list numpy`
  - install a specific version with `conda install numpy=1.13`

# 3 - Why Use NumPy?
Faster with math
  - plain python took 9.31 seconds to run an operation
  - numpy took 0.092 seconds
Handles multidimensional array objects
  - represent vectors and matrices
Optimized for matrix operations

# 4 - Creating and Saving NumPy ndarrays
Creating numpy arrays (ndarray)
  - `x = np.array([1,2,3,4,5])`: created from an existing regular python list
  - can also build them using built-in NumPy functions
  - x.dtype lets you know the type of the data in the array (not the same as the class of the data)
ndaaray
  - 1 dimensional == rank 1 array
  - shape: size along each of its dimensions
  - ndarrays have attributes that allow us to get info in an intuitive way (e.g. `.shape` => `(5,))

```
x = np.array([1, 2, 3, 4, 5])

x.shape has dimensions: (5,)
type(x) is an object of type: class 'numpy.ndarray'
x.dtype The elements in x are of type: int64
```

NumPy Data Types
  - ndarrays can also hold strings
  - all elements must be the same type
  - if you try to create ndarray from list with mixed data types, all will be translated as strings

After creating an ndarray you can save it to a file to use later
  - `x = np.array([1,2,3,4,5])`
  - `np.save('my_array', x)` => saved into file in this directory called `my_aray.npy)`
  - `y = np.load('my_array.npy')`

# 5 - Using Built-in Functions to Create ndarrays
Use built in functions to create ndarrays
  ```
  # We create a 3 x 4 ndarray full of zeros. 
  X = np.zeros((3,4))
```

`np.ones()` works similarly but fills the array with 1s

  ```
  # We create a 2 x 3 ndarray full of fives. 
  X = np.full((2,3), 5) 
  ```

Identity Matrix
  - square matrix that has only 1s in its main diagonal and zeros everywhere else

  ```
  # We create a 5 x 5 Identity matrix. 
  X = np.eye(5)
  X =
  [[ 1. 0. 0. 0. 0.]
   [ 0. 1. 0. 0. 0.]
   [ 0. 0. 1. 0. 0.]
   [ 0. 0. 0. 1. 0.]
   [ 0. 0. 0. 0. 1.]]
  ```

  ```
  # Create a 4 x 4 diagonal matrix that contains the numbers 10,20,30, and 50
  # on its main diagonal
  X = np.diag([10,20,30,50])
  ```

```
np.arange(start,stop,step)
  - start is inclusive
  - stop is exclusive
  - step is the distance between values in the array
# We create a rank 1 ndarray that has sequential integers from 0 to 9
x = np.arange(10)
x = [0 1 2 3 4 5 6 7 8 9]
```

```
linspace - start and stop points are inclusive unless you pass in endpoint = false

# We create a rank 1 ndarray that has 10 integers evenly spaced between 0 and 25.
x = np.linspace(0,25,10)
```

```
np.reshape(ndarray, new_shape)
```
  - new shape should be compatable with number of elements in original array

  ```
  x = np.arange(20)
  [ 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19]

  x = np.reshape(x, (4,5))
  [[ 0 1 2 3 4]
   [ 5 6 7 8 9]
   [10 11 12 13 14]
   [15 16 17 18 19]]
  ```

Can be chained:
`np.arange(20).reshape(4,5)`

```
# We create a 3 x 3 ndarray with random floats in the half-open interval [0.0, 1.0).
X = np.random.random((3,3))
[[ 0.12379926 0.52943854 0.3443525 ]
 [ 0.11169547 0.82123909 0.52864397]
 [ 0.58244133 0.21980803 0.69026858]]
```

```
# We create a 3 x 2 ndarray with random integers in the half-open interval [4, 15).
X = np.random.randint(4,15,size=(3,2))
[[ 7 11]
 [ 9 11]
 [ 6 7]]
```

```
# We create a 1000 x 1000 ndarray of random floats drawn from normal (Gaussian) distribution
# with a mean of zero and a standard deviation of 0.1.
X = np.random.normal(0, 0.1, size=(1000,1000))
```

# 6 - Quiz: Create an ndarray
```
import numpy as np

# Using the Built-in functions you learned about in the
# previous lesson, create a 4 x 4 ndarray that only
# contains consecutive even numbers from 2 to 32 (inclusive)

X = np.linspace(2,32,16).reshape(4,4)
```

# 7 - Accessing, Deleting, and Inserting Elements into ndarrays
ndarrays are:
  - mutable
  - able to be sliced
Accessing elements in the array
  - positive numbers in square brackets from 0th to nth
  - negative numbers in square brackets from nth to 0th
  - access 2d array elements in the same way x[0,0]
  * NOTE: [0,0] refers to the element in the first row, first column
Modify values the way you usually do in an array
  - `x[2]` = 7 `[0,1,7,3,4]`
Adding and Deleting Elements
deleting with .delete
```
# We create a rank 1 ndarray 
x = np.array([1, 2, 3, 4, 5])
x = np.delete(x, [0,4])
x => [2 3 4]
```

```
# We create a rank 2 ndarray
Y = np.array([[1,2,3],[4,5,6],[7,8,9]])

# We delete the first row of y
w = np.delete(Y, 0, axis=0)
w =
[[4 5 6]
 [7 8 9]]

# We delete the first and last column of y
v = np.delete(Y, [0,2], axis=1)
v =
[[2]
 [5]
 [8]]
```

adding with .append
```
# We create a rank 1 ndarray 
x = np.array([1, 2, 3, 4, 5])
# We append the integer 6 to x
x = np.append(x, 6)
x = [1 2 3 4 5 6]
```

```
# We create a rank 2 ndarray 
Y = np.array([[1,2,3],[4,5,6]])
# We append a new row containing 7,8,9 to y
v = np.append(Y, [[7,8,9]], axis=0)
v =
[[1 2 3]
 [4 5 6]
 [7 8 9]]

# We append a new column containing 9 and 10 to y
q = np.append(Y,[[9],[10]], axis=1)
q =
[[ 1 2 3 9]
 [ 4 5 6 10]]
```

inserting a value into an array
`np.insert(ndarray, index, elements, axis)`

```
# We create a rank 1 ndarray 
x = np.array([1, 2, 5, 6, 7])
# We insert the integer 3 and 4 between 2 and 5 in x. 
x = np.insert(x,2,[3,4])
x => [1,2,3,4,5,6,7]
```

```
# We create a rank 2 ndarray 
Y = np.array([[1,2,3],[7,8,9]])
# We insert a row between the first and last row of y
w = np.insert(Y,1,[4,5,6],axis=0)
w =
[[1 2 3]
 [4 5 6]
 [7 8 9]]

# We insert a column full of 5s between the first and second column of y
v = np.insert(Y,1,5, axis=1)
v =
[[1 5 2 3]
 [7 5 8 9]]
```

Allowed to stack ndarrays on top of each other
```
# We create a rank 1 ndarray 
x = np.array([1,2])

# We create a rank 2 ndarray 
Y = np.array([[3,4],[5,6]])

# We stack x on top of Y
z = np.vstack((x,Y))
z =
[[1 2]
 [3 4]
 [5 6]]

# We stack x on the right of Y. We need to reshape x in order to stack it on the right of Y. 
w = np.hstack((Y,x.reshape(2,1)))
w =
[[3 4 1]
 [5 6 2]]

```

# 8 - Slicing ndarrays
1. ndarray[start:end]
  - select elements between start and end
2. ndarray[start:]
  - select elements from start until last index
3. ndarray[:end]
  - select all elements from first index until end

[Link to Review](https://classroom.udacity.com/nanodegrees/nd089/parts/8de94dee-7635-43b3-9d11-5e4583f22ce3/modules/dd3e4af8-d576-427e-baae-925fd16ff2ff/lessons/1b5143bb-1d0f-49d7-a3c1-a87c5307260d/concepts/b267fa21-ebe2-4a73-b7b5-3c09c82782ff)

NOTE: Slices provide a narrowed view into an ndarray but they still point to the same array. If you make changes to the slice, you will make changes to the original array.
  - append `.copy()` to the slice to make sure the variable you're assigning to is a new array

`np.diag(X)` grab elements on the main diagonal of the array
`np.diag(X, k=1)` grab elements above the main diagonal
`np.diag(X, k=-1)` grab elements below the main diagonal

`np.unique()` gives you a rank 1 array with unique elements in the ndarray

# 9 - Boolean Indexing, Set Operations, and Sorting
Helps you figure out which indices you want when you don't know

```
# We create a 5 x 5 ndarray that contains integers from 0 to 24
X = np.arange(25).reshape(5, 5)
X =
[[ 0 1 2 3 4]
 [ 5 6 7 8 9]
 [10 11 12 13 14]
 [15 16 17 18 19]
 [20 21 22 23 24]]
print('The elements in X that are greater than 10:', X[X > 10])
The elements in X that are greater than 10: [11 12 13 14 15 16 17 18 19 20 21 22 23 24]
```

You can also use the booleans to assign new values to any index that fits the criteria

```
# We use Boolean indexing to assign the elements that are between 10 and 17 the value of -1
X[(X > 10) & (X < 17)] = -1
X =
[[ 0 1 2 3 4]
 [ 5 6 7 8 9]
 [10 -1 -1 -1 -1]
 [-1 -1 17 18 19]
 [20 21 22 23 24]]
```

`np.intersect1d(x,y))` - The elements that are both in x and y
`np.setdiff1d(x,y))` - The elements that are in x that are not in y
`np.union1d(x,y))` - All the elements of x and y

`np.sort(x))`
  - sorts the elements in the array
  - does not mutate the original array
  - could also use `x.sort()`
  - when sorting rank 2 arrays, you need to specify whether you're sorting by rows or columns

```
X =
[[6 1 7 6 3]
  [3 9 8 3 5]
  [6 5 8 9 3]
  [2 1 5 7 7]
  [9 8 1 9 8]]
np.sort(X, axis = 0))
[[2 1 1 3 3]
  [3 1 5 6 3]
  [6 5 7 7 5]
  [6 8 8 9 7]
  [9 9 8 9 8]]
np.sort(X, axis = 1))
[[1 3 6 6 7]
  [3 3 5 8 9]
  [3 5 6 8 9]
  [1 2 5 7 7]
  [1 8 8 9 9]]
```

# 10 - Quiz: Manipulating ndarrays
```
import numpy as np

# Create a 5 x 5 ndarray with consecutive integers from 1 to 25 (inclusive).
# Afterwards use Boolean indexing to pick out only the odd numbers in the array

# Create a 5 x 5 ndarray with consecutive integers from 1 to 25 (inclusive).
X = np.arange(1,26).reshape(5,5)

# Use Boolean indexing to pick out only the odd numbers in the array
Y = X[X % 2 != 0]

```

# 11 - Arithmetic Operations and Broadcasting
NumPy allows the following on ndarrays:
  - element-wise operations 
  - matrix operations
Broadcasting
  - how NumPy handles element-wise arithmetic operations with ndarrays of different shapes
  - used implicitly when doing arithmetic operations between scalars and ndarrays
Addition
  - `np.add()` or `+`
  * NOTE: function allows you to tweak operations with keywords
Subtraction
  - `np.subtract()` or `-`
  * NOTE: function allows you to tweak operations with keywords
Multiplication
  - `np.multiply()` or `*`
  * NOTE: function allows you to tweak operations with keywords
Division
  - `np.divide()` or `/`
  * NOTE: function allows you to tweak operations with keywords
Exponentiation
  - `np.exp()`
  - `np.sqrt()`
  - `np.power(x,2)`
Statistics (can pass in axis=0 or 1)
  - `X.mean()` - average
  - `X.sum()` - sum
  - `X.std()` - standard deviation
  - `X.median()` - median
  - `X.max()` - max value
  - `X.min()` - min value
Math on all of the elements in the array:
  - `3 * X`
  - `3 + X`
  - `3 - X`
  - `3 / X`
All of the above work the same on rank 2 matrices


# 12 - Quiz: Creating ndarrays with Broadcasting
```
import numpy as np

# Use Broadcasting to create a 4 x 4 ndarray that has its first
# column full of 1s, its second column full of 2s, its third
# column full of 3s, etc.. 
ones_array = np.ones((4,4))
nums = [1,2,3,4]

X = ones_array * nums
```

# 13 - Getting Set Up for the Mini-Project
All set to work through it with the embedded notebooks.

# 14 - Mini-Project: Mean Normalization and Data Separation
normalizing / feature scaling
  - ensure all data is at the same scale
mean normalization
  - distribute values evenly in some small range around 0
After the data has been mean normalized, it is customary in machine learnig to split our dataset into three sets:
  - Training Set: 60% of data
  - Cross Validation Set: 20% of data
  - Test Set: 20% of data
`np.random.permutation(N)` 
  - creates a random permutation of integers from 0 to N - 1
  - this is handy if you know that you have N rows and you want to do a random sampling
    - use `.permutation` to get the indices of the rows
    - my_array[perm_results]
