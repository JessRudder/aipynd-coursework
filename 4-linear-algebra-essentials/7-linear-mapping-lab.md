# 1 - Lab Description
First: Visualize process of matrix multiplication with simple example in Python

Second: Solve more complex problem using matrix multiplication in Python

# 2 - Visualizing Matrix Multiplication
In this lab you will:
  - Graph a vector decomposed into it's basis vectors  𝑖̂   and  𝑗̂  
  - Graph a vector transformation that uses Equation 1
  - Demonstrate that the same vector transformation can be achieved with matrix multiplication (Equation 2)

Using numpy's matrix multiplication to multiply some matrix by some basis:
```
# Define vector v 
v = np.array([-1,2])

# Define 2x2 matrix ij
ij = np.array([[3, 1],[1, 2]])

# TODO 1.: Demonstrate getting v_trfm by matrix multiplication
# by using matmul function to multiply 2x2 matrix ij by vector v
# to compute the transformed vector v (v_t) 
v_t = np.matmul(v, ij)
```

# 3 - Matrix Multiplication Lab
Goal is to create a matrix of all your currencies and what you could convert them into using some matrix multiplication:

Make your intiial currency matrix (a 1x8)
```
import numpy as np
import pandas as pd

# Creates numpy vector from a list to represent money (inputs) vector.
money = np.asarray([70, 100, 20, 80, 40, 70, 60, 100])

# Creates pandas dataframe with column labels(currency_label) from the numpy vector for printing.
currency_label = ["USD", "EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "HKD"]
money_df = pd.DataFrame(data=money, index=currency_label, columns=["Amounts"])
print("Inputs Vector:")
money_df.T
```

Make your currency matrix (8x8)
```
# Sets path variable to the 'path' of the CSV file that contains the conversion rates(weights) matrix.
path = %pwd

# Imports conversion rates(weights) matrix as a pandas dataframe.
conversion_rates_df = pd.read_csv(path+"/currencyConversionMatrix.csv",header=0,index_col=0)

# Creates numpy matrix from a pandas dataframe to create the conversion rates(weights) matrix.
conversion_rates = conversion_rates_df.values

# Prints conversion rates matrix.
print("Weights Matrix:")
conversion_rates_df
```

Multiply them together to make your how much money will I have matrix (1x8)
```
# TODO 1.: Calculates the money totals(outputs) vector using matrix multiplication in numpy.
money_totals = np.matmul(money, conversion_rates)


# Converts the resulting money totals vector into a dataframe for printing.
money_totals_df = pd.DataFrame(data = money_totals, index = currency_label, columns = ["Money Totals"])
print("Outputs Vector:")
money_totals_df.T
```

# 4 - Linear Mapping Lab Solution
Videos of each of the problems and their solutions.
