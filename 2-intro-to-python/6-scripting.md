# 1 - Introduction
How to set up and use a local Python development environment

# 2 - Python Installation
Check that python is installed (it is both 2 and 3)

# 3 - Install Python Using Anaconda
Install Anaconda for data science stuff
  - distribution of libraries and software for data science


# 4 - [For Windows] Configuring Git Bash to Run Python
Configuring git bash for Windows

# 5 - Running a Python Script
use `python3` command to run script
  - e.g. `python3 script.py`

# 6 - Programming Environment Setup
Choose and use and IDE or notepad dev environment.

# 7 - Editing a Python Script
Prove that you can edit and run a python script

# 8 - Scripting with Raw Input
Get raw input from user using `input` method
  - `name = input("Enter your name: ")`
    - name is what the var will be called
    - "Enter your name: " is the prompt that will be shown
Interpret user input using `eval`
```
result = eval(input("Enter an expression: "))
```

# 9 - Quiz Scripting with Raw Input
Had us write a script

# 10 - Solution: Scripting with Raw Input
Showed us what they wrote.

# 11 - Errors and Exceptions
`syntax errors` - Python can't interpret the code
`exceptions` - unexpected things happen during execution of the program

# 12 - Quiz: Errors and Exceptions
Checking understanding of errors and exceptions. Went through different types of possible exceptions.

# 13 - Handling Errors
`try` - begins the try statement
`except` - if an exception occurs this block will be run (can specify for specific exception types)
`else` - runs if there are no exceptions after running `try`
`finally` - runs no matter what after try, except, else (whether errors occurred or not)

# 14 - Practice: Handling Input Errors
Had us write a `try` `except` block with a specific error for zero division.

# 15 - Solution: Handling Input Errors
No new information

# 16 - Accessing Error Messages
You have access to an error message in an except block

```
try:
    # some code
except ZeroDivisionError as e:
   # some code
   print("ZeroDivisionError occurred: {}".format(e))
```

for catchall exceptions

```
try:
    # some code
except Exception as e:
   # some code
   print("Exception occurred: {}".format(e))
```

# 17 - Reading and Writing Files
reading to a file
```
f = open('my_path/my_file.txt', 'r')
file_data = f.read()
f.close()
```
writing to a file
```
f = open('my_path/my_file.txt', 'w')
f.write("Hello there!")
f.close()
```

Another option that auto closes the file when it's done

```
with open('my_path/my_file.txt', 'r') as f:
    file_data = f.read()
```

It's possible to open too many files at once....so, be careful.

# 18 - Quiz: Reading and Writing Files
```
def create_cast_list(filename):
    cast_list = []
    #use with to open the file filename
    #use the for loop syntax to process each line
    #and add the actor name to cast_list
    with open(filename, 'r') as f:
        for line in f:
            actor = line.split(",")[0]
            cast_list.append(actor)

    return cast_list
```

# 19 - Solution: Reading and Writing Files
No new information


# 20 - Quiz: Practice Debugging
Did it! Needed a string of a number to be an int.

# 21 - Solutions for Quiz: Practice Debugging
No new information

# 22 - Importing Local Scripts
Imports give you access to 'modules' (e.g. the code in other files)

```
import useful_functions
useful_functions.add_five([1, 2, 3, 4])
```

You can give it an alias:

```
import useful_functions as uf
uf.add_five([1, 2, 3, 4])
```

IMPORTANT NOTE:
To avoid running executable statements in a script when it's imported as a module in another script, include these lines in an `if __name__ == "__main__"` block. Or alternatively, include them in a function called `main()` and call this in the if main block.

# 23 - The Standard Library
The classes, methods and functions that are avaialable to you when you install Python.
  - all the things you get for free!

# 24 - Quiz: The Standard Library
Questions about different modules

# 25 - Solution: The Standard Library
Here are some important modules to know:

`csv`: very convenient for reading and writing csv files
`collections`: useful extensions of the usual data types including `OrderedDict`, `defaultdict` and `namedtuple`
`random`: generates pseudo-random numbers, shuffles sequences randomly and chooses random items
`string`: more functions on strings. This module also contains useful collections of letters like `string.digits` (a string containing all characters which are valid digits).
`re`: pattern-matching in strings via regular expressions
`math`: some standard mathematical functions
`os`: interacting with operating systems
`os.path`: submodule of os for manipulating path names
`sys`: work directly with the Python interpreter
`json`: good for reading and writing json files (good for web work)

# 26 - Techniques for Importing Modules
Different variations on imports
  - import individual function/class
    `from module_name import object_name`
  - import multiple individuals
    `from module_name import first_object, second_object`
  - rename module
    `import module_name as new_name`
  - import an object from a module and rename it
    `from module_name import object_name as new_name`
  - import every object individual from a module (DO NOT DO THIS)
    `from module_name import *`
  - standard import for when you want all the objects
    `import module_name`
  - import a submodule like this
    `import package_name.submodule_name`

# 27 - Quiz: Techniques for Importing Modules
No new information

# 28 - Third-Party Libraries
`pip` is the built in package manager for Pythin
  - `pip install package_name`
Use a `requirements.txt` file for larger projects
  - like bundler (sorta)

ex:
```
beautifulsoup4==4.5.1
bs4==0.0.1
pytz==2016.7
requests==2.11.1
```

Then install with a single line like so:

`pip install -r requirements.txt`

Then listed a bunch of useful third party libraries

# 29 - Experimenting with an Interpreter
Can just use default one with `python3`
There's a fancier one called `IPython`

# 30 - Online Resources
Suggestion for how to search for help with code errors

Include Python resources ordered by reliability
  - top is [Python Tutorial](https://docs.python.org/3/tutorial/)

# 31 - Practice Question
Worked through writing code to do the following:
- create a dictionary of letters and flower names
- ask a person for their first and last name
- print a message saying the flower that has the same first name is...

# 32 - Solution for Practice Question
No new information

# 33 - Conclusions
You have skills to tackle scripting in Python now
  - Congratulations!
