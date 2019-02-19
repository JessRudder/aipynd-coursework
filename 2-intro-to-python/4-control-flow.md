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
`for` loops are used to iterate over something
`iterable` is object that can return one of its elements at a time

```
cities = ['new york city', 'mountain view', 'chicago', 'los angeles']
for city in cities:
    print(city)
```

Common pattern is give iteration variable and iterable same names

`range()` 
  - exclusive of the upper bounds
  * `range(3)` will run 3 times
  - `range(start=0, stop, step=1)`


Iterating over a collection when you need to keep track of the index in the collection
```
for index in range(len(cities)):
    cities[index] = cities[index].title()
```

# 11 - Practice: For Loops
```
for num in range(5, 31, 5):
    print(num)
```

# 12 - Solution: For Loops Practice
No new information provided

# 13 - Quiz: For Loops
Wrote some more for loops


# 14 - Solution: For Loops Quiz
No new information

# 15 - Quiz: Match Inputs to Outputs
Checked for knowledge on for loops and ranges

# 16 - Building Dictionaries
Using list of keys and for loops to build a dictionary
  - remember to check if key exists before assinging it (or you'll just overwrite things)
```
for word in book_title:
    if word not in word_counter:
        word_counter[word] = 1
    else:
        word_counter[word] += 1
```

or

```
for word in book_title:
    word_counter[word] = word_counter.get(word, 0) + 1
```

# 17 - Iterating through Dictionaries with For Loops
If you iterate through in the normal way you will only get access to the keys


Get access to key and values in iteration:
```
for key, value in cast.items():
    print("Actor: {}    Role: {}".format(key, value))
```

# 18 - Quiz: Iterating Through Dictionaries with For Loops

```
result = 0
basket_items = {'apples': 4, 'oranges': 19, 'kites': 3, 'sandwiches': 8}
fruits = ['apples', 'oranges', 'pears', 'peaches', 'grapes', 'bananas']

#Iterate through the dictionary

#if the key is in the list of fruits, add the value (number of fruits) to result
for item, count in basket_items.items():
    if item in fruits:
        result += count
```

# 19 - Solution: Iterating Through Dictionaries with For Loops
No new information

# 20 - While Loops
Starts with `while` then a conditional that should continue looping until it evaluates to false

```
while sum(hand)  < 17:
    hand.append(card_deck.pop())
```

`sum()` returns sum of elements in a list
`pop()` removes last element from a list and returns it

# 21 - Practice: While Loops
Did some for and while loops

# 22 - Solution: While Loops Practice
No new information

# 23 - Quiz: While Loops
```
limit = 40
nearest_square = 0
current_square = 0
counter = 3
# write your while loop here
while current_square < limit:
    nearest_square = current_square
    current_square = counter * counter
    counter += 1
```

# 24 - Solution: While Loops Quiz
```
limit = 40

num = 0
while (num+1)**2 < limit:
    num += 1
nearest_square = num**2
```

# 25 - For Loops vs While Loops
`for` when you know how many iterations you want to run
`while` when you know what condition you want to meet

# 26 - Check for Understanding: For and While Loops
You need to make sure the while loop has:
  - a condition expression that will be assessed and when met, will allow you to exit the loop
  - make sure the loop is advancing
  - the value of the condition variables is changing with each iteration of the loop.


# 27 - Solution: Check for Understanding: For and While Loops
```
num_list = [422, 136, 524, 85, 96, 719, 85, 92, 10, 17, 312, 542, 87, 23, 86, 191, 116, 35, 173, 45, 149, 59, 84, 69, 113, 166]

odd_count = 0
idx = 0
odd_sum = 0

while (odd_count < 5) and (idx < len(num_list)):
    if not (num_list[idx] % 2) == 0:
        odd_sum += num_list[idx]
        odd_count += 1
    idx += 1
```

# 28 - Break, Continue
`break` terminates a loop
`continue` skips one iteration of a loop

# 29 - Quiz: Break, Continue
```
news_ticker = ""
# write your loop here
for headline in headlines:
    if len(news_ticker) + len(headline) <= 140:
        news_ticker = news_ticker + headline
    else:
        news_ticker = news_ticker + headline[:(140-len(news_ticker))]
        break
    
    if len(news_ticker) >= 140:
        break
    news_ticker += " "
    

print(news_ticker)
```

# 30 - Solution: Break, Continue
```
for headline in headlines:
    news_ticker += headline + " "
    if len(news_ticker) >= 140:
        news_ticker = news_ticker[:140]
        break

print(news_ticker)
```

# 31 - Practice: Loops
```
for num in check_prime:
    for i in range(2, num):
        if num % i == 0:
            print("{} is NOT a prime number".format(num))
            break

    if i == num - 1:
        print("{} IS a prime number".format(num))
```

# 32 - Solution: Loops
No new information

# 33 - Zip and Enumerate
`zip` returns an iterator that combines multiple iterables into one sequence of tuples
  - Each tuple contains the elements in that position from all the iterables.
  `list(zip(['a', 'b', 'c'], [1, 2, 3]))` => `[('a', 1), ('b', 2), ('c', 3)]`

Unzip with an asterisk
```
some_list = [('a', 1), ('b', 2), ('c', 3)]
letters, nums = zip(*some_list)
```

`enumerate` is a built in function that returns an iterator of tuples containing indices and values of a list
  - You'll often use this when you want the index along with each element of an iterable in a loop
  ```
  letters = ['a', 'b', 'c', 'd', 'e']
  for i, letter in enumerate(letters):
    print(i, letter)
  ```

# 34 - Quiz: Zip and Enumerate
```
for coords in zip(labels, x_coord, y_coord, z_coord):
    points.append("{}: {}, {}, {}".format(coords[0], coords[1], coords[2], coords[3]))


for point in points:
    print(point)
```

zip into a dict!
`cast = dict(zip(cast_names, cast_heights))`


# 35 - Solution: Zip and Enumerate
No new information


# 36 - List Comprehensions
Concise way to make lists

```
capitalized_cities = []
for city in cities:
    capitalized_cities.append(city.title())
```

can be:

`capitalized_cities = [city.title() for city in cities]`

Steps
  - start with brackets `[]`
  - expression to evaluate for each element in the iterable
  - the iterable (e.g. `city in cities`)
  - optional conditional after the iterable
  `squares = [x**2 for x in range(9) if x % 2 == 0]`
  - can even have an else!
  `squares = [x**2 if x % 2 == 0 else x + 3 for x in range(9)]`

# 37 - Quiz: List Comprehensions
```
names = ["Rick Sanchez", "Morty Smith", "Summer Smith", "Jerry Smith", "Beth Smith"]

first_names = [name.split()[0].lower() for name in names]
```

```
scores = {
             "Rick Sanchez": 70,
             "Morty Smith": 35,
             "Summer Smith": 82,
             "Jerry Smith": 23,
             "Beth Smith": 98
          }

passed = [name for name, score in scores.items() if score >= 65]
```

# 38 - Solution: List Comprehensions
No new information

# 39 - Practice Questions
```
most_win_director = []
win_count_dict = {}
# Add your code here
for year, winnerlist in winners.items():
    for winner in winnerlist:
        win_count_dict[winner] = win_count_dict.get(winner, 0) + 1

current_most = 0

for director, num_wins in win_count_dict.items():
    if num_wins > current_most:
        most_win_director = [director]
        current_most = num_wins
    elif num_wins == current_most:
        most_win_director.append(director)
```

# 40 - Solution to Practice Questions
Close to their solution, but there were a few differences

```
highest_count = 0
most_win_director = []

for key, value in win_count_dict.items():
    if value > highest_count:
        highest_count = value
        most_win_director.clear()
        most_win_director.append(key)
    elif value == highest_count:
        most_win_director.append(key)
    else:
        continue
```

They use `.clear()` to clear out the list where we just reset it to the value of `[author]`

They have an `else` with `continue` to make it explicit that it should go to the next loop

Alternative, compact solution:
```
highest_count = max(win_count_dict.values())

most_win_director = [key for key, value in win_count_dict.items() if value == highest_count]
```

# 41 - Conclusion
Next you'll learn how to organize your code into functions!
