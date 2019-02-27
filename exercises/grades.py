names =  input("Please enter names separated by commas: ") # get and process input for a list of names
assignments =  input("Please enter assignment counts separated by commas: ") # get and process input for a list of the number of assignments
grades =  input("Please enter grades separated by commas: ") # get and process input for a list of grades

# message string to be used for each student
# HINT: use .format() with this string in your for loop
message = "Hi {},\n\nThis is a reminder that you have {} assignments left to \
submit before you can graduate. You're current grade is {} and can increase \
to {} if you submit all assignments before the due date.\n\n"

# write a for loop that iterates through each set of names, assignments, and grades to print each student's message
name_list = names.title().split(",")
assignment_list = assignments.split(",")
grade_list = grades.split(",")

for name, assignment, grade in zip(name_list, assignment_list, grade_list):
    print(message.format(name, assignment, grade, int(grade) + int(assignment)*2))
