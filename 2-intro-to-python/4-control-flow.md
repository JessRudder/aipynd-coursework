# 1 - Introduction
Add more functionality to your code with control flow:
  - conditional statements
  - boolean expressions
  - for and while loops
  - break and continue
  - zip and enumerate
  - list comprehensions

# 2 - Conditional Statements
`if` statement
  - conditional statement that runs or skips code based on whether condition is true/false
  ```
  if phone_balance < 5:
    phone_balance += 10
    bank_balance -= 10
  ```
    - starts with `if` followed by `condition to check` and a `:`
      - needs to be boolean expression that evals to `True` or `False`
    - indented block of code to be executed if condition is true

Use comparison operators in conditional statements
`==` test for equality
`!=` test for non equality

If, Elif, Else
`elif` can extend `if` clause with another boolean test (followed by code to execute if it's true)

`else` is fallback code to run if none of the other branches eval to `True`

Indentation
Python uses indentation to enclose blocks of code (vs `{}` as in many other languages)

Indents conventionally come in multiples of four spaces
  - can vary by team, but whole team should be consistent
  - cannot mix spaces and tabs

# 3 - Practice: Conditional Statements
Practiced filling in some `if`, `elif` statements


# 4 - Solution: Conditional Statements
You will only fall through to a later `elif` statement if the ones before it have evaluated to `False`


# 5 - Quiz: Conditional Statements
Practiced filling in some `if`, `elif` statements

# 6 - Solution: Conditional Statements
No additional information

# 7 - Boolean Expressions for Conditions
Complex Boolean Expressions
  - you may need a more complex comparison with multiple comparison/logical operators/calculations
  ```
  if 18.5 <= weight / height**2 < 25:
      print("BMI is considered 'normal'")

  if is_raining and is_sunny:
      print("Is there a rainbow?")

  if (not unsubscribed) and (location == "USA" or location == "CAN"):
      print("send email")
  ```
  * simple or complex, the `if` condition must be a boolean expression that evals to `True` or `False`

Good and Bad Examples
  - Don't use `True` or `False` as conditionals
  - Be Careful writing expressions that use logical operators
  - Don't compare boolean variable with `== True` or `== False`

Truth Value Testing
If you use a non-boolean object as a conditional Python will check it's truth value
  - `None`, `False`, `0`, `""`, `[]`, `()`, etc are `False`
  - Most other objects are considered `True`


# 8 - Quiz: Boolean Expressions for Conditions
Adjusted some `if` statements we saw earlier to use variable assignment with them

# 9 - Solution: Boolean Expressions for Conditions
Remember to avoid unnecessarily wordy conditionals
`if prize:` instead of `if prize == None` (then have the None case handled in the else)

# 10 - For Loops


# 11 - Practice: For Loops


# 12 - Solution: For Loops Practice


# 13 - Quiz: For Loops


# 14 - Solution: For Loops Quiz


# 15 - Quiz: Match Inputs to Outputs


# 16 - Building Dictionaries


# 17 - Iterating through Dictionaries with For Loops


# 18 - Quiz: Iterating Through Dictionaries with For Loops


# 19 - Solution: Iterating Through Dictionaries with For Loops


# 20 - While Loops


# 21 - Practice: While Loops


# 22 - Solution: While Loops Practice


# 23 - Quiz: While Loops


# 24 - Solution: While Loops Quiz


# 25 - For Loops vs While Loops


# 26 - Check for Understanding: For and While Loops


# 27 - Solution: Check for Understanding: For and While Loops


# 28 - Break, Continue


# 29 - Quiz: Break, Continue


# 30 - Solution: Break, Continue


# 31 - Practice: Loops


# 32 - Solution: Loops


# 33 - Zip and Enumerate


# 34 - Quiz: Zip and Enumerate


# 35 - Solution: Zip and Enumerate


# 36 - List Comprehensions


# 37 - Quiz: List Comprehensions


# 38 - Solution: List Comprehensions


# 39 - Practice Questions


# 40 - Solution to Practice Questions


# 41 - Conclusion

