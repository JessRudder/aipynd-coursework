# 1 - Introduction
You're gonna learn how to group these data types into data structures

Final two operator types
  - membership
  - identity

# 2 - Lists and Membership Operators
data structures - containers that organize and group data types together in different ways
list
  - mutable ordered sequence of elements
  - defined using square brackets
  - can be a mix of the data types we've seen before
  - can look up individual elements based on their index
  - zero index
  - can index from the end using negative (-1 is last element)
  - if you attempt to access and index that doesn't exist in the list, you will get `IndexError: list index out of range`

slice and dice with lists
slicing
  - pull more than one value by providing where to start and where to end
    - lower: inclusive
    - upper: exclusive
  - short cuts
    - omit starting index to start at beginning of the list
    - omit ending index to go to end of list
membership operator
  - `in` checks if item is in list or string and returns a bool
    - `'in' in 'this is a string' => True`
    - `5 in [1, 2, 3] => False`
  - `not in` checks if item is not in a list or string and returns a bool
    - `'but' not in 'butts' => False`
    - `5 not in [1,4,5] => False`

mutability and order
  - strings are sequences of letters and lists are any type of objects
  - lists can be modified but strings can't (immutable)

Mutability - whether or not we can change an object once it has been created

Order - whether the position of an element in the object can be used to access the element

# 3 - Quiz: Lists and Membership Operators
```
month = 8
days_in_month = [31,28,31,30,31,30,31,31,30,31,30,31]

# use list indexing to determine the number of days in month
num_days = days_in_month[month-1]

print(num_days)
```

```
eclipse_dates = ['June 21, 2001', 'December 4, 2002', 'November 23, 2003',
                 'March 29, 2006', 'August 1, 2008', 'July 22, 2009',
                 'July 11, 2010', 'November 13, 2012', 'March 20, 2015',
                 'March 9, 2016']
                 
                 
# TODO: Modify this line so it prints the last three elements of the list
print(eclipse_dates[-3:])
```

# 4 - Solution: List and Membership Operators
Nothing new here.

# 5 - Why Do We Need Lists?
Hold collections of information (such as all of the stock symbols in a specific fund)
  - be able to manipulate that data
  - be able to check for membership in that collection

# 6 - List Methods
  - `len()` returns how many elements are in a list
  - `max()` returns greatest element of the list (depending on types of elements)
    - numbers: largest number
    - strings: last in an alpha sorted list
    - mixed lists: undefined
  - `min()` returns smallest element in the list (opposite of `max()`)
  - `sorted()` returns. copy of the list in order from smallest to largest (leaving original list unchanged)
  - `join()` takes list of strings and returns a string consisting of list elements joined by a separator string
    - e.g. `new_str = " ".join(["fore", "aft", "starboard", "port"]) => "fore aft starboard port"`
  - `append()` adds an element to the end of the list

# 7 - Quiz: List Methods
Testing the methods above
  - no new information

# 8 - Check for Understanding: Lists
data type:
  - type that classifies data
  - can include primitive data types (e.g. integers, booleans, strings) or data structures (e.g. lists)

data structures:
  - containers that organize and group data types together in different ways

# 9 - Tuples
Used to store related pieces of information (e.g. latitude and longitude)
  - ordered
  - immutable

Can be used to assign multiple variables in a compact way

```
dimensions = 52, 40, 100
length, width, height = dimensions
print("The dimensions are {} x {} x {}".format(length, width, height))
```

2nd line is `tuple unpacking` where multiple variables are assigned fromt he content of the tuple dimensions

If you don't need to use the var `dimensions` you can just assign as follows:

```
length, width, height = 52, 40, 100
```

Note: parenthesis are optional when defining tuples and programmers frequently omit them if the parentheses don't clarify the code

# 10 - Quiz: Tuples
Tuples: ordered and immutable
Lists: ordered and mutable

# 11 - Sets
Sets
  - mutable, unordered collection of unique elements
  - one use: quickly remove duplicates from a list

```
numbers = [1, 2, 6, 3, 1, 1, 6]
unique_nums = set(numbers)
print(unique_nums)
=> {1, 2, 3, 6}
```

  - supports `in` operator
  - uses `add` instead of `append`
  - remove with `pop()` but it will remove a random element
  - can do mathematical set operations
    - `.union()`
    - `.intersection()`
    - `.difference()`

# 12 - Quiz: Sets
No new information

# 13 - Dictionaries and Identity Operators


# 14 - Quiz: Dictionaries and Identity Operators


# 15 - Solution: Dictionaries and Identity Operators


# 16 - Quiz: More With Dictionaries


# 17 - When to Use Dictionaries?


# 18 - Check for Understanding: Data Structures


# 19 - Compound Data Structures


# 20 - Quiz: Compound Data Structures


# 21 - Solution: Compound Data Structures


# 22 - Practice Questions


# 23 - Solution: Practice Questions


# 24 - Conclusion
