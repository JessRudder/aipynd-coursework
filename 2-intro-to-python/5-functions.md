# 1 - Introduction
Will cover:
  - Defining Functions
  - Variable Scope
  - Documentation
  - Lambda Expressions
  - Iterators and Generators

# 2 - Defining Functions
```
def cylinder_volume(height, radius):
    pi = 3.14159
    return height * pi * radius ** 2
```

and call

`cylinder_volume(10, 3)`

Often includes a `return` statement but doesn't have to

Naming conventions:
  - letters, numbers, underscores (no spaces)
  - can't use reserved words or built-in identifiers
  - use descriptive names

Default arguments
`def cylinder_volume(height, radius=5)`
  - height has to be specified but radius will default to 5 if it's not passed in

Can pass in arguments by position or by name

# 3 - Quiz: Defining Functions
Wrote some functions

# 4 - Solution: Defining Functions
```
def readable_timedelta(days):
    # use integer division to get the number of weeks
    weeks = days // 7
    # use % to get the number of days that remain
    remainder = days % 7
    return "{} week(s) and {} day(s).".format(weeks, remainder)
```

# 5 - Check for Understanding: Functions
A function associated with an object is called a `method`

# 6 - Variable Scope
Variable scope
  - which parts of a program a variable can be referenced, or used, from
  - variables created in a function can only be used inside that function
  - variables defined outside of functions can be accessed globally
    * if you want access to modify it, you have to pass it in as an argument

Best Practice - define variables in the smallest scope needed (prevent collisions!!!!!!!!!)

# 7 - Quiz: Variable Scope
Testing modifying a global variable inside a function - UnboundLocalError

# 8 - Solution: Variable Scope
No new information

# 9 - Check for Understanding: Variable Scope
No new information

# 10 - Documentation
Docstrings are a type of comment used to explain the purpose of a function and how it should be used:

```
def population_density(population, land_area):
    """Calculate the population density of an area. """
    return population / land_area
```
  - first line brief explanation of functions purpose
  - if you feel more is needed, you can add it after the one line summary

# 11 - Quiz: Documentation
Write your own docstring for a method.

# 12 - Solution: Documentation
Their example:
```
"""
Return a string of the number of weeks and days included in days.

Parameters:
days -- number of days to convert (int)

Returns:
string of the number of weeks and days included in days
"""
```

# 13 - Lambda Expressions
Anonymous functions

Take this:

```
def multiply(x, y):
    return x * y
```

and turn it into this:

`multiply = lambda x, y: x * y`

Call them both the same way:

`multiply(4, 7)`

Components of a Lambda Function
  - The lambda keyword is used to indicate that this is a lambda expression.
  - Following lambda are one or more arguments for the anonymous function separated by commas, followed by a colon :. Similar to functions, the way the arguments are named in a lambda expression is arbitrary.
  - Last is an expression that is evaluated and returned in this function. This is a lot like an expression you might see as a return statement in a function.

# 14 - Quiz: Lambda Expressions
Wrote some lambdas

# 15 - Solution: Lambda Expressions
No new information

# 16 - Iterators and Generators
`iterables` are objects that can return one of their elements at a time, such as a list
`iterators` are an object that represents a stream of data
`generators` are simple way to create iterators using functions
  - check out the keyword `with` in Python

```
def my_range(x):
    i = 0
    while i < x:
        yield i
        i += 1
```

```
for x in my_range(5):
    print(x)
```

=>

```
0
1
2
3
4
```

Generators are a lazy way to build iterables when the fully realized list would not fit in memory or cost to calculate it would be high.

# 17 - Quiz: Iterators and Generators
```
def chunker(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]


for chunk in chunker(range(25), 4):
    print(list(chunk))
```

# 18 - Solution: Iterators and Generators
No new information

# 19 - Generator Expressions
Combine generators and list comprehensions
  - create a generator similar to how you create a list comprehension but with parenthesis instead of brackets

```
sq_list = [x**2 for x in range(10)]  # this produces a list of squares

sq_iterator = (x**2 for x in range(10))  # this produces an iterator of squares
```

# 20 - Conclusion
[talk on writing functions](https://youtu.be/rrBJVMyD-Gs)
[yield blog post](https://jeffknupp.com/blog/2013/04/07/improve-your-python-yield-and-generators-explained/)
