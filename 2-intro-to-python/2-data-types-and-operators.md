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

# 12 - Solution: Booleans, Comparison Operators and Logical Operators
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

# 13 - Strings
Immutable ordered series of characters
Can be created with single or double quotes
  - There are a few edge cases
What if you want quotation marks in your string?
  - Place string in single quotes rather than double quotes (so you can use the other one in the string)
  - Use backslash to escape the quotes that should be part of the string and not an ending quote
String operators:
  + put strings together
  * repeat strings
Use `len()` to determine how many characters are in the string  

# 14 - Quiz: Strings
```
given_name = "William"
middle_names = "Bradley"
family_name = "Pitt"

name_length = len(given_name + " " + middle_names + " " + family_name) #todo: calculate how long this name is

# Now we check to make sure that the name fits within the driving license character limit
# Nothing you need to do here
driving_license_character_limit = 28
print(name_length <= driving_license_character_limit)
```

Using len() on an integer will return this error `TypeError: object of type 'int' has no len()`

# 15 - Solution: Strings
Another alternative:
```
# TODO: Fix this string!
ford_quote = "Whether you think you can, or you think you can't--you're right."
```

```
name_length = len(given_name) + len(middle_names) + len(family_name) + 2
```

```
len("{0} {1} {2}".format(given_name, middle_names, family_name))
```

# 16 - Type and Type Conversion
Four data types we've seen so far:
  - int
  - float
  - bool
  - string
Check types using `type()`
  - e.g. type('this')

# 17 - Quiz: Type and Type Conversion
```
mon_sales = "121"
tues_sales = "105"
wed_sales = "110"
thurs_sales = "98"
fri_sales = "95"

#TODO: Print a string with this format: This week's total sales: xxx
# You will probably need to write some lines of code before the print statement.

sales_total = int(mon_sales) + int(tues_sales) + int(wed_sales) + int(thurs_sales) + int(fri_sales)

print("This week's total sales: " + str(sales_total))
```

# 18 - Solution: Type and Type Conversion

Nothing too radical here. Just what you had above really.

# 19 - String Methods
Methods are functions that belong to an object (in this case a string)
  - `lower()` => `sample_string.lower()`

`format()` allows you to format your strings
  `print("Mohammed has {} balloons".format(27))` => `Mohammed has 27 balloons`

```
  animal = "dog"
  action = "bite"
  print("Does your {} {}?".format(animal, action)) => Does your dog bite?
```

  - `"{0} {1} {2}".format(given_name, middle_names, family_name)`

# 20 - Quiz: String Methods
```
# Write two lines of code below, each assigning a value to a variable
person = "I"
action = "move it move it"

# Now write a print statement using .format() to print out a sentence and the 
#   values of both of the variables
print("{0} like to {1}!".format(person, action))
```

# 21 - Another String Method - Split
`split()` returns a list that contains the words from the input string

Has two additional arguments
  - `sep`: stands for "separator" and can identify how the string should be split up (e.g. whitespace, comma, etc) => defaults to whitespace
  - `maxsplit`: provides the maximum number of splits - resulting in maxsplit + 1 number of arguments in the new list (remaining string is last item returned)

# 22 - Quiz: String Methods Practice
```
verse = "If you can keep your head when all about you\n  Are losing theirs and blaming it on you,\nIf you can trust yourself when all men doubt you,\n  But make allowance for their doubting too;\nIf you can wait and not be tired by waiting,\n  Or being lied about, don’t deal in lies,\nOr being hated, don’t give way to hating,\n  And yet don’t look too good, nor talk too wise:"
print(verse)

# Use the appropriate functions and methods to answer the questions above
# Bonus: practice using .format() to output your answers in descriptive messages!

print("length: " + str(len(verse)))
print("first and index: " + str(verse.find("and")))
print("last you index: " + str(verse.rfind("you")))
print("count of you: " + str(verse.count("you")))
```

# 23 - Solution: String Methods Practice
Could get fancy with things:
```
message = "Verse has a length of {} characters.\nThe first occurence of the \
word 'and' occurs at the {}th index.\nThe last occurence of the word 'you' \
occurs at the {}th index.\nThe word 'you' occurs {} times in the verse."

length = len(verse)
first_idx = verse.find('and')
last_idx = verse.rfind('you')
count = verse.count('you')

print(message.format(length, first_idx, last_idx, count))
```

# 24 - There's a Bug in my Code
Debugging tips:
  - understand common error messages (and what to do with them)
  - search for error message using web community
  - use print statements

Understanding Common Error Messages
  - `"ZeroDivisionError: division by zero." ` - you can't divide by zero
  - `"SyntaxError: unexpected EOF while parsing"` - often produced when you've left out something (like a parenthesis), means it got to the end of file and didn't find what it expected
  - `"TypeError: len() takes exactly one argument (0 given)"` - the number of arguments that are expected were not present when the method was called

Search for Your Error Message
  - copy and paste error message into web browser
  - search using keywords from error or situation you're facing

Use Print Statements to Help Debugging
  - add temporary print statements to help you see which lines were executed before the error occurs

# 25 - Conclusion

We've got building blocks....we're gonna learn how to piece them together.

# 26 - Summary
Read the stuff I wrote above, fool!

What's Next
  - Data Structures!

Additional Practice Resources
  - hackerrank
  - codewars
