# 1 - introduction
People learn at different rates
  - take your time to understand
Follow along by typing in code editor under video in each example
Print function will be common tool to see what's going on

# 2 - arithmetic operators
print(3 + 5) # will print output of that operation

+ addition
- subtraction
* multiplication
/ division
** exponent
^ bitwise XOR
% modulo (returns remainder from division)
// integer division (answer rounded down to integer, even if answer is negative -3.5 => -4)

python follows the mathmatical order of operations

# 3 - Quiz: arithmetic operators

# Write an expression that calculates the average of 23, 32 and 64
// Place the expression in this print statement
print((23+32+64)/3)

// Fill this in with an expression that calculates how many tiles are needed.
print((9*7)+(5*7))

// Fill this in with an expression that calculates how many tiles will be left over.
print((17*6) - ((9*7) + (5*7)))

# 4 - Solution: Arithmetic operators
Prefer spaces around + and - operators but not * and /
Prefer no space between method call and ()

# 5 - Variables and Assignment Operators
Variables I
  Prefer snake case for var names
  mv_population = 74728

  = operator that assigns value on right to what is on the left

  After assignment, you can use a variables name to access the value it holds
  y = 2
  print(y)
  => 2
Variables II
  Python allows multiple assignment
    x = 3
    y = 4
    z = 5

    could also be

    x, y, z = 3, 4, 5

  Variable names should be descriptive of the values they represent

  Things to keep in mind
    - only use letters, numbers and underscores (no spaces)
    - must start with letter or underscore
    - avoid reserved words or built-in identifiers
    - pythonic way is to use snake case (snake_case)

Assignment Operators
  mv_population = mv_population + 4000 - 600

  could be

  mv_population += 4000 - 600

  =
  +=
  -=
  *=
  /=

# 6 - Quiz: Variables and Assignment Operators
// The current volume of a water reservoir (in cubic metres)
reservoir_volume = 4.445e8
// The amount of rainfall from a storm (in cubic metres)
rainfall = 5e6

// decrease the rainfall variable by 10% to account for runoff
rainfall *= .9

// add the rainfall variable to the reservoir_volume variable
reservoir_volume += rainfall

// increase reservoir_volume by 5% to account for stormwater that flows
// into the reservoir in the days following the storm
reservoir_volume *= 1.05

// decrease reservoir_volume by 5% to account for evaporation
reservoir_volume *= .95

// subtract 2.5e5 cubic metres from reservoir_volume to account for water
// that's piped to arid regions.
reservoir_volume -= 2.5e5

// print the new value of the reservoir_volume variable
print(reservoir_volume)

# 7 - Solution: Variables and Assignment Operators
When a variable is assigned, it is assigned to the value of the expression on the right-hand-side, not to the expression itself

# 8 - Integers and Floats
Integers and Floats
  Two data types that can be used for numeric values:
    - int: for integer values
    - float: fr decimal or floating point values
    - x = int(4.7) => 4
    - y = float(4) => 4.0
  Check types using the type function
    - print(type(x)) => int
    - print(type(y)) => float
  Floats are approximations which are slightly more than 0.1
    - necessary because floats can represent a huge range of numbers
    - this can be seen with floating point math
    - print(.1 + .1 + .1 == .3) => False

Python Best Practices
  print(4 + 5)
  not
  print(     4 + 5)

  limit each line of code to 80 characters (though 99 is fine for certain uses)

# 9 - Quiz: Integers and Floats
Error you see when attempting to divide by zero
  ```
  Traceback (most recent call last):
    File "/tmp/vmuser_oghrpjvrvo/quiz.py", line 1, in <module>
      print(5/0)

  ZeroDivisionError: division by zero
  ```
Two types of errors to look for:
  - Exceptions: problem that occurs when code is running
  - Syntax: problem detected when python checks the code before it runs

# 10 - Booleans, Comparison Operators and Logical Operators
Bool:
  - used to represent the values True or False
  - often encoded as (1 or 0)
Comparison Operators:
  <
  >
  <=
  >=
  ==
  !=
Logical Operators:
  and - checks if both sides are true
  or - checks if at least one side is true
  not - flips the bool value

# 11 - Quiz: Booleans, Comparison Operators and Logical Operators
  sf_population, sf_area = 864816, 231.89
  rio_population, rio_area = 6453682, 486.5

  san_francisco_pop_density = sf_population/sf_area
  rio_de_janeiro_pop_density = rio_population/rio_area

  // Write code that prints True if San Francisco is denser than Rio, and False otherwise
  print(san_francisco_pop_density > rio_de_janeiro_pop_density)

12 - Solution: Booleans, Comparison Operators and Logical Operators
could be
```
print(san_francisco_pop_density > rio_de_janeiro_pop_density)
```
or
```
if (san_francisco_pop_density > rio_de_janeiro_pop_density):
    print (True)
else:
    print (False)
```
