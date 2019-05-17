# 1 - Vectors Lab
Going to learn:
  - Plotting a 2D vector
  - Multiplying a 2D vector by a scalar and plotting the results
  - Adding two 2D vectors together and plotting the results

To plot a 2D vector:
1. Make both NumPy and Matlibplot python packages available using the import method
2. Define vector  𝑣⃗   
3. Plot vector  𝑣⃗   using Matlibplot
  - Create a variable ax to reference the axes of the plot
  - Plot the origin as a red dot at point 0,0 using ax and plot method
  - Plot vector  𝑣⃗   as a blue arrow with origin at 0,0 using ax and arrow method
  - Format x-axis
    - Set limits using xlim method
    - Set major tick marks using ax and set_xticks method
  - Format y-axis
    - Set limits using ylim method
    - Set major tick marks using ax and set_yticks method
  - Create the gridlines using grid method
  - Display the plot using show method

```
# Import NumPy and Matplotlib
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

# Define vector v 
v = np.array([1,1])

# Plots vector v as blue arrow with red dot at origin (0,0) using Matplotlib

# Creates axes of plot referenced 'ax'
ax = plt.axes()

# Plots red dot at origin (0,0)
ax.plot(0,0,'or')

# Plots vector v as blue arrow starting at origin 0,0
ax.arrow(0, 0, *v, color='b', linewidth=2.0, head_width=0.20, head_length=0.25)

# Sets limit for plot for x-axis
plt.xlim(-2,2)

# Set major ticks for x-axis
major_xticks = np.arange(-2, 3)
ax.set_xticks(major_xticks)


# Sets limit for plot for y-axis
plt.ylim(-1, 2)

# Set major ticks for y-axis
major_yticks = np.arange(-1, 3)
ax.set_yticks(major_yticks)

# Creates gridlines for only major tick marks
plt.grid(b=True, which='major')

# Displays final plot
plt.show()
```

Scaling a vector and plotting it:
```
# Define vector v 
v = np.array([1,1])

# Define scalar a
a = 3

# TODO 1.: Define vector av - as vector v multiplied by scalar a
av = v*a

# Plots vector v as blue arrow with red dot at origin (0,0) using Matplotlib

# Creates axes of plot referenced 'ax'
ax = plt.axes()

# Plots red dot at origin (0,0)
ax.plot(0,0,'or')

# Plots vector v as blue arrow starting at origin 0,0
ax.arrow(0, 0, *v, color='b', linewidth=2.5, head_width=0.30, head_length=0.35)

# TODO 2.: Plot vector av as dotted (linestyle='dotted') vector of cyan color (color='c') 
# using ax.arrow() statement above as template for the plot 
ax.arrow(0, 0, *av, color='c', linewidth=2.5, head_width=0.30, head_length=0.35, linestyle='dotted')


# Sets limit for plot for x-axis
plt.xlim(-2, 4)

# Set major ticks for x-axis
major_xticks = np.arange(-2, 4)
ax.set_xticks(major_xticks)


# Sets limit for plot for y-axis
plt.ylim(-1, 4)

# Set major ticks for y-axis
major_yticks = np.arange(-1, 4)
ax.set_yticks(major_yticks)

# Creates gridlines for only major tick marks
plt.grid(b=True, which='major')

# Displays final plot
plt.show()
```

Key part of plotting vector addition:
```
# Plots vector v as blue arrow starting at origin 0,0
ax.arrow(0, 0, *v, color='b', linewidth=2.5, head_width=0.30, head_length=0.35)

# Plots vector w as cyan arrow with origin defined by vector v
ax.arrow(v[0], v[1], *w, linestyle='dotted', color='c', linewidth=2.5, 
         head_width=0.30, head_length=0.35)
```
  - vector w is ploted with vector v's origin as it's own origin (the first two params passed in to .arrow)

# 2 - Vectors Lab Solution
Walks through the solutions.
