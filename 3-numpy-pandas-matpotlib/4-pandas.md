# 1 - Instructors
Same people from the previous section
Pandas
  - data manipuation and analysis tool

# 2 - Introduction to pandas
Name comes from "panel data"
  - econometrics term
Gives Python 2 additional data structures that allow us to work with labeled and relational data in easy/intuitive manner
  - Pandas Series
  - Pandas DataFrame
Course was created using Pandas 0.22
  - `conda install pandas=0.22`

# 3 - Why Use pandas?
Machine Learning relies on large amounts of data
  - it's important that the data is high quality
  - Pandas is designed for fast data analysis and manipulation
Features:
  - Allows the use of labels for rows and columns
  - Can calculate rolling statistics on time series data
  - Easy handling of NaN values
  - Is able to load data of different formats into DataFrames
  - Can join and merge different datasets together
  - It integrates with NumPy and Matplotlib


# 4 - Creating pandas Series
Pandas series
  - one-dimensional array-like object that can hold many data types
  - unlike ndarrays, you can assign index label to each element in the Pandas series

`import pandas as pd`
  - common to import it as `pd`

Create a Pandas Series
  - `pd.Series(data, index)`
  - `index` is a list of index labels

```
# We import Pandas as pd into Python
import pandas as pd

# We create a Pandas Series that stores a grocery list
groceries = pd.Series(data = [30, 6, 'Yes', 'No'], index = ['eggs', 'apples', 'milk', 'bread'])

# We display the Groceries Pandas Series
groceries
```
```
eggs           30
apples         6
milk         Yes
bread       No
dtype: object
```

Indices are in the first column
Data is in the second column

Similar methods that we had with ndarray
  - `.shape`: shape of array
  - `.ndim`: dimensions
  - `.size`: number of elements
  - `.values`: data stored in array
  - `.index`: index labels for array

Check if an index label exists with `in` command


# 5 - Accessing and Deleting Elements in pandas Series
Access elements in pandas series
  - numerical index inside []
  - labels provided on creation in []
  - `.loc` makes it explicit that we're using a labeled index
  - `.iloc` makes it explicit that we're using an integer index
Pandas Series are Mutable
  - change items like this `groceries['eggs'] = 2`
  - delete items with `.drop()`
    - returns new array
  - delete (in place) with `.drop(label, inplace = True)`
    - changes the original array

# 6 - Arithmetic Operations on pandas Series
You can run arithmetic on elements in a pandas series as long as the operation you're using can be run on all data types in the series
  - `fruits + 2` will add 2 to every item in the list
  - `fruits[0] + 2` will add 2 to the 0th element in the list
  - `np.exp(fruits)`
  - `np.sqrt(fruits)`
  - `np.power(fruits, 2)`

# 7 - Quiz: Manipulate a Series
```
import pandas as pd

distance_from_sun = [149.6, 1433.5, 227.9, 108.2, 778.6]

planets = ['Earth','Saturn', 'Mars','Venus', 'Jupiter']

dist_planets = pd.Series(data = distance_from_sun, index = planets)

time_light = dist_planets / 18

close_planets = time_light[time_light < 40]
```

# 8 - Creating pandas DataFrames
Pandas DataFrames
  - 2 dimensional data structures with labeled rows and columns
  - can hold many data types
  - similar to an Excel spreadsheet
  - can be created manually or with impoted data from a file

```
# We import Pandas as pd into Python
import pandas as pd

# We create a dictionary of Pandas Series 
items = {'Bob' : pd.Series(data = [245, 25, 55], index = ['bike', 'pants', 'watch']),
         'Alice' : pd.Series(data = [40, 110, 500, 45], index = ['book', 'glasses', 'bike', 'pants'])}

# We create a Pandas DataFrame by passing it a dictionary of Pandas Series
shopping_carts = pd.DataFrame(items)

# We display the DataFrame
shopping_carts

        Alice     Bob
bike    500.0     245.0
book    40.0      NaN
glasses 110.0     NaN
pants   45.0      25.0
watch   NaN       55.0
 ```

Pandas inserts `NaN` anywhere that a value was not provided

Has the same set of functions available to give info on the DataFrame
  - `.shape`
  - `.ndim`
  - `.size`
  - `.values`: provides just the data without row/column labels
  - `.index`: gives the row index labels
  - `.columns`: gives the column index labels

Can create with just a subset of data:
```
# We Create a DataFrame that only has selected items for Alice
alice_sel_shopping_cart = pd.DataFrame(items, index = ['glasses', 'bike'], columns = ['Alice'])
```
This will create a frame with only the Alice colun and only the glasses/bike rows

Can also create a DataFrame using a list of python dictionaries:
```
# We create a list of Python dictionaries
items2 = [{'bikes': 20, 'pants': 30, 'watches': 35}, 
          {'watches': 10, 'glasses': 50, 'bikes': 15, 'pants':5}]

# We create a DataFrame  and provide the row index
store_items = pd.DataFrame(items2, index = ['store 1', 'store 2'])
```

# 9 - Accessing Elements in pandas DataFrames
Many ways to access:
  - rows
  - columns
  - individual elements using row/column labels

```
store_items[['bikes']]
store_items[['bikes', 'pants']]
store_items.loc[['store 1']]
store_items['bikes']['store 2']
```

When acessing individual elements, you should always provide the [column] then the [row]

```
store_items['shirts'] = [15,2]
```

This creates a shirts column with a value of 15 for row 1 and 2 for row 2

To add a row, create a new Dataframe and append it to the original DataFrame:

```
# We create a dictionary from a list of Python dictionaries that will number of items at the new store
new_items = [{'bikes': 20, 'pants': 30, 'watches': 35, 'glasses': 4}]

# We create new DataFrame with the new_items and provide and index labeled store 3
new_store = pd.DataFrame(new_items, index = ['store 3'])

# We append store 3 to our store_items DataFrame
store_items = store_items.append(new_store)
```

Appending a new row to the dataframe results in the columns being alphabetized

Add column anywhere you want:
```
# We insert a new column with label shoes right before the column with numerical index 4
store_items.insert(4, 'shoes', [8,5,0])
```

Deleting data
  - `.pop` removes columsn
  ```
    # We remove the new watches column
    store_items.pop('new watches')
  ```
  - `.drop` can remove rows or columns depending on which axis you give it
  ```
    # We remove the watches and shoes columns
    store_items = store_items.drop(['watches', 'shoes'], axis = 1)

    # We remove the store 2 and store 1 rows
    store_items = store_items.drop(['store 2', 'store 1'], axis = 0)
  ```

Change row/column labels with `.rename()`

```
# We change the column label bikes to hats
store_items = store_items.rename(columns = {'bikes': 'hats'})

# We change the row label from store 3 to last store
store_items = store_items.rename(index = {'store 3': 'last store'})
```

Change the index to be one of the columns in the DataFrame:
```
# We change the row index to be the data in the pants column
store_items = store_items.set_index('pants')
```

# 10 - Dealing with NaN
You need a way to detect/correct errors in your data
  - most common bad data is missing values (NaN)

Figure out the number of NaN values in your dataframe
```
# We count the number of NaN values in store_items
x =  store_items.isnull().sum().sum()
```
  - `.isnull()` returns a Boolean DataFrame the same size as `store_items` with `True` where you previously had `NaN`
  - `True` == 1 so if you count the `True`s using `.sum()`
  - First `.sum()` gives sums across colmns
  - Run `.sum()` again to get the total number of `NaN`s across everything

Determine the number of NaN values with `.count()`
  - my_data_frame.count()

You can delete or replace the `NaN` values:
Eleminate rows/columns that contain NaN
  - `.dropna(axis)` eleminates rows with `NaN` when `axis = 0` and columns with `NaN` when `axis = 1`
  - this drops out of place, so, if you want to modify the original dataframe, you have to pass in `inplace = True`
Replace `NaN` values with `.fillna()`
  - `.fillna(0)` replaces them with 0
  - `.fillna(method = 'ffill', axis)` uses `forward filling` where the previous value is used to replace a NaN
  - `.fillna(method = 'backfill', axis)` replaces NaNs with the data that comes after them in the DataFrame
  - `.interpolate(method = 'linear', axis)` will use linear interpolation to replace `NaN` values using the given axis

NOTE: For all of these, you have to pass in `inplace = True` if you want it to modify the existing DataFrame

# 11 - Quiz: Manipulate a DataFrame
```
# Now replace all the NaN values in your DataFrame with the average rating in
# each column. Replace the NaN values in place. HINT: you can use the fillna()
# function with the keyword inplace = True, to do this. Write your code below:

book_ratings.fillna(book_ratings.mean(), inplace = True)
```

# 12 - Loading Data into a pandas DataFrame
Most common data storage format is CSV
  - load CSV files into Pandas DataFrames using `pd.read_csv()`

```
# We load Google stock data in a DataFrame
Google_stock = pd.read_csv('./GOOG.csv')
```

You can get a look at the first 5 rows with `Google_stock.head()`

You can get a look at the last 5 rows with `Google_stock.tail()`

You can also pass a number of rows that you'd like to see in to `.head()` or `.tail()`

Quick check for null columns:
  - `Google_stock.isnull().any()`

`.describe()` will give descriptive statistics on each column of the DataFrame
  - count
  - mean
  - std
  - min
  - 25%
  - 50%
  - 75%
  - max

```
# We get descriptive statistics on a single column of our DataFrame
Google_stock['Adj Close'].describe()
```

Can also get just a single stat with .max(), .min(), etc

Tell if the data in different columns are correlated using `.corr()`
  - 1 tells us there is a high correlation
  - 0 tells us it's not correlated at all

`.groupby()` method allows us to group data in different ways

```
# We display the total amount of money spent in salaries each year
data.groupby(['Year'])['Salary'].sum()
```

```
# We display the salary distribution per department per year.
data.groupby(['Year', 'Department'])['Salary'].sum()
```
This will group by department by year then show the sum of salaries for that department in that year

# 13 - Getting Set Up for the Mini-Project
Gonna run it in the hosted JuPyTer notebooks

# 14 - Mini-Project: Statistics From Stock Data

