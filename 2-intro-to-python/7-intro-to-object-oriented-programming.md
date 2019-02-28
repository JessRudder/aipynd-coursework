# 1 - Introduction
  - allows you to organize code for large programs
  - hides implementation from the end-user
  - provides an example from `scikit-learn`

# 2 - Procedural vs Object-Oriented Programming
`procedural` - running blocks of code based on method names
`object-oriented` - a collection of characteristics and actions

# 3 - Class, Object, Method and Attribute
`class` - a blueprint of methods and attributes
`object` - instance of a class
`attribute` - descriptor or characteristic
`method` - an action that a class or object could take
`oop` - common abbreviation of Object-Oriented Programming
`encapsulation` - combine functions and data into a single entity (in oop it's called a class)

# 4 - OOP Syntax
How to write a class in Python syntax

```
class Name:
  def __init__(self, shirt_color, blah):
    self.color = shirt_color
    self.blah = blah

  def change_price(self, new_price):
    self.price = new_price
```

`new_shirt = Shirt('red',.....)`

self - a reference to the instantiated object

# 5 - Exercise: OOP Syntax Practice - Part 1
Instantiate an object, change attributes, call methods

# 6 - A Couple of Notes about OOP
Python prefers to change attributes directly (often frowned upon in other languages)

`shirt_one.price = 10`

Can optionally write getters and setters:

```
def set_price(self, new_price):
      self._price = new_price
```

Underscore before a variable name indicates that the attribute is meant to be private
  - isn't enforced but is indication that you shouldn't directly access it
  - `shirt_one._price = 10` == NO!

Advantages to using methods instead of direct access
  - hide implementation details from end user

Modularized Code
  - put Shirt class in its own Python script (shirt.py)
  - in another Python script, import the class with line like `from shirt import Shirt`

# 7 - Exercise: OOP Syntax Practice - Part 2
Created a couple more objects and methods

Note: You keep messing up by not passing `self` as the first argument in a method where you need access to self

# 8 - Commenting Object-Oriented Code
Commenting helps people understand the "why" of what's going on in the code
  - sometimes it's nice to explain the "how" too no matter what those obnoxious Ruby devs tell you

Use docstrings
  - we learned about them earlier

# 9 - A Gaussian Class
Formulas for Gaussian distribution and Binomial distribution
  - if these are required knowledge, you should probably go back and review them

# 10 - How the Gaussian Class Works
This is a description of how the Gaussian Class works
  - walks through methods, etc

# 11 - Exercise: Code the Gaussian Class
Write methods to do Gaussian things in a Gaussian class

# 12 - Magic Methods
What happens when you add two gaussian distributions together
  - add means of individual Gaussian dists together (to get mean of new distribution)
  - std dev of new dist takes square root of sum of squares of the standard deviations

If you tried to add them together with python, it wouldn't know what to do

Magic methods allow you to override built in methods to give your class a custom method

`add` = built in method
`__add__` = magic method

You can override quite a bit of built in methods (many of them math)

# 13 - Exercise: Code Magic Methods


# 14 - Inheritance
Giving a class a bunch of methods just because it came from another class

`class Clothing:` (setting up a parent class)

```
class Shirt(Clothing):
  def __init__(self, color, size,...):
    Clothing.__init__(self, color,...)
    self.long_or_short = long_or_short
```

# 15 - Exercise: Inheritance with Clothing
Practiced building an inherited class

Practiced adding methods to child and parent

# 16 - Inheritance: Probability Distribution
Another look at inheritance with the Gaussian Distribution class

Showed overwriting to read data file

# 17 - Demo: Inheritance Probability Distributions
Get a fancy look at the Gaussian Distribution class in action!

# 18 - Advanced OOP Topics
We've learned the basics of object orientation, but, there are advanced topics too!
  - `class methods`, `instance methods`, and `static methods` - these are different types of methods that can be accessed at the class or object level
  - `class attributes` vs `instance attributes` - you can also define attributes at the class level or at the instance level
  - `multiple inheritance`, `mixins` - A class can inherit from multiple parent classes
  - `Python decorators` - Decorators are a short-hand way for using functions inside other functions
