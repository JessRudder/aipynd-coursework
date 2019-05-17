# 1 - Linear Combinations
span of 2 vectors - set of all possible vectors that you can reach with a linear combination of a given pair of vectors

If t is in span of v and w, it can be written as a linear combination of them:
𝑎𝑣⃗ +𝑏𝑤⃗ =𝑡⃗  ,  where  𝑣⃗   and  𝑤⃗   are multiplied by scalars  𝑎  and  𝑏  and then added together to produce vector  𝑡⃗  .
  - if you find a pair of values that makes the equation true, t is in the span of vw
  - otherwise t is not in their span

Steps to Determine a Vector's Span Computationally:
1. Make the NumPy Python package available using the import method
2. Create right and left sides of the augmented matrix
  - Create a NumPy vector  𝑡⃗   to represent the right side of the augmented matrix.
  - Create a NumPy matrix named  𝑣𝑤  that represents the left side of the augmented matrix ( 𝑣⃗   and  𝑤⃗  )
3. Use NumPy's linalg.solve function to check a vector's span computationally by solving for the scalars that make the equation true. For this lab you will be using the check_vector_span function you will defined below.

Note the Following:
  - linalg.solve function will only solve for the scalars (vector_of_scalars) that will make equation 1 true, ONLY if the vector that's being checked (vector_to_check) is within the span of the other vectors (set_of_vectors).
  - Otherwise, the vector (vector_to_check) is not within the span and an empty vector is returned.

`vector_of_scalars = np.linalg.solve(vw, t)`

vw is the span we want to check
t is the vector we are looking for in the span

System of Equations
𝑎𝑣⃗ +𝑏𝑤⃗ =𝑡⃗ could be written as:

Single Solution
x|1| + y|2| = | 4|
 |3|    |5|   |11|

x  + 2y = 4   Solve x = 2, y = 1
3x + 5y = 11

```
%matplotlib inline
import matplotlib.pyplot as plt
plt.plot([4,0],[0,2],'b',linewidth=3)
plt.plot([3.6667,0],[0,2.2],'c-.',linewidth=3)
plt.plot([2],[1],'ro',linewidth=3)
plt.xlabel('Single Solution')
plt.show()
```

Infinite Solutions:
x|1| + y|2| = | 6|
 |2|    |4|   |12|

These are the same line so there are infinite values that could solve this.

```
import matplotlib.pyplot as plt
plt.plot([6,0],[0,3],'b',linewidth=5)
plt.plot([1,4,6,0],[2.5,1,0,3],'c-.',linewidth=2)
plt.xlabel('Redundant Equations')
plt.show()
```

No Solution:
x|1| + y|2| = | 6|
 |1|    |2|   |10|

Nothing could ever make the above true

```
import matplotlib.pyplot as plt
plt.plot([10,0],[0,5],'b',linewidth=3)
plt.plot([0,6],[3,0],'c-.',linewidth=3)
plt.xlabel('No Solution')
plt.show()
```


# 2 - Linear Combination Lab Solution
Videos showing how to solve.
