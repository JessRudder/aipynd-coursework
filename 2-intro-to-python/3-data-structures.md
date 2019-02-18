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
Dictionary
  - mutable data type that stores mappings of unique keys to values
  - like a hash in Ruby (I'm sure there are reasons this is wrong, but, it's oh so right)
  - `elements = {"hydrogen": 1, "helium": 2, "carbon": 6}`
  * keys can be any immutable type (string, integers, tuples, etc)
  - look up/insert values using square brackets that enclose the key
    - e.g. `elements["helium"]` and `elements["lithium"] = 3`
  - `in` checks values in dictionary 
    - e.g. "carbon" in elements
  - `get` looks up values in dictionary
    - e.g. elements.get("dilithium")
    * if key is not found, returns `None` (this is different from the square brackets which will fail if the key doesn't exist)
    - can have a default value instead of `None` if the key does not exist `elements.get('kryptonite', 'There\'s no such element!')`
Identity Operators
  - `is` evaluates if both sides have the same identity
  - `is not` evaluates if both sides have different identities
  ```
  n = elements.get("dilithium")
  n is None => True
  n is not None => False
  ```


# 14 - Quiz: Dictionaries and Identity Operators
```
population = {
    "Shanghai": 17.8,
    "Istanbul": 13.3,
    "Karachi": 13.0,
    "Mumbai": 12.5
}
```

`str`, `int` and `float` can also be keys to dictionaries

If you use bracket lookup notation and the key is not in the dictionary, a `KeyError` occurs

`==` checks for equality
`is` checks for identity

```
a = [1, 2, 3]
b = a
c = [1, 2, 3]
```

`a == b` and `a is b`
`a == c` but `a is not b`

# 15 - Solution: Dictionaries and Identity 
You can put dictionary all on one line or each of the elements on it's own line. It's a stylistic choice.


# 16 - Quiz: More With Dictionaries
Tested understanding of indexing on dictionaries

# 17 - When to Use Dictionaries?
Collecting, storing and working with more information than simple strings or integers

Works well when there'sa linkage between the data being stored that can be broken down to key/value pairs

# 18 - Check for Understanding: Data Structures
`tuple` is an immutable, ordered data structure that can be indexed and sliced like a list

`sets` and `dictionaries` are both defined with `{}`
  - `sets` are a sequence of elements separated by commas
  - `dictionary` is sequence of key/value pairs, marked w/colons and separated by commas
  - `a = {}` will default to empty dict
  - `set()` and `dict()` will make empty sets and dicts


# 19 - Compound Data Structures
Contain containers in other containers to create compound data structures

```
elements = {"hydrogen": {"number": 1,
                         "weight": 1.00794,
                         "symbol": "H"},
              "helium": {"number": 2,
                         "weight": 4.002602,
                         "symbol": "He"}}
```

```
helium = elements["helium"]  # get the helium dictionary
hydrogen_weight = elements["hydrogen"]["weight"]  # get hydrogen's weight
```

# 20 - Quiz: Compound Data Structures
```
elements = {'hydrogen': {'number': 1, 'weight': 1.00794, 'symbol': 'H'},
            'helium': {'number': 2, 'weight': 4.002602, 'symbol': 'He'}}

# todo: Add an 'is_noble_gas' entry to the hydrogen and helium dictionaries
# hint: helium is a noble gas, hydrogen isn't

elements['helium']['is_noble_gas'] = True
elements['hydrogen']['is_noble_gas'] = False
```

# 21 - Solution: Compound Data Structures
No new information

# 22 - Practice Questions
```
verse_dict =  {'if': 3, 'you': 6, 'can': 3, 'keep': 1, 'your': 1, 'head': 1, 'when': 2, 'all': 2, 'about': 2, 'are': 1, 'losing': 1, 'theirs': 1, 'and': 3, 'blaming': 1, 'it': 1, 'on': 1, 'trust': 1, 'yourself': 1, 'men': 1, 'doubt': 1, 'but': 1, 'make': 1, 'allowance': 1, 'for': 1, 'their': 1, 'doubting': 1, 'too': 3, 'wait': 1, 'not': 1, 'be': 1, 'tired': 1, 'by': 1, 'waiting': 1, 'or': 2, 'being': 2, 'lied': 1, 'don\'t': 3, 'deal': 1, 'in': 1, 'lies': 1, 'hated': 1, 'give': 1, 'way': 1, 'to': 1, 'hating': 1, 'yet': 1, 'look': 1, 'good': 1, 'nor': 1, 'talk': 1, 'wise': 1}
print(verse_dict, '\n')

# find number of unique keys in the dictionary
num_keys = len(verse_dict.keys())
print(num_keys)

# find whether 'breathe' is a key in the dictionary
contains_breathe = verse_dict.get('breathe')
print(contains_breathe)

# create and sort a list of the dictionary's keys
sorted_keys = sorted(set(verse_dict.keys()))

# get the first element in the sorted list of keys
print(sorted_keys[0])

# find the element with the highest value in the list of keys
print(max(verse_dict, key=verse_dict.get)) 
```

# 23 - Solution: Practice Questions
`print(sorted_keys[-1]) ` was answer for final one. I read it wrong. I thought it meant the key with the highest value (out of key value pairs). It meant the key that had the highest index in the sorted list.

# 24 - Conclusion
Good understanding of data structures is important for programming and data analysis
  - data analysts must understand which data types and data structures are available (and when to use them)
A dictionary is mutable but it's keys must be immutable
