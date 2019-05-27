# 1 - Instructor
Mat Leonard
  - going over our first project in the School of AI
  - Image Classifier

# 2 - Project Intro
It will become more important to incorporate deep learning models into everyday applications
  - image classification on cameras
  - prevalence of voice assistants

This project:
  - create an image classifier to classify some images
  - start by developing model in Jupyter notebook
  - then convert code to python application that runs in the command line

# 3 - Introduction to GPU Workspaces
We'll be using special workspaces that have GPU support enabled
  - sessions are connections from browser to remote server
  - students have limited number of GPU hours allocated on server
  - workspace data stored in user's home folder is preserved between sessions (up to 3gb)
  - workspace sessions are preserved if connection drops or window is closed
  - workspace sessions are automatically terminated after a period of inactivity
  - kernel state is preserved as long as notebook session remains open

NOTE: If GPU mode is active, it will remain active after closing or stopping a notebook. You can only stop GPU mode with the GPU toggle button

Toggling GPU Mode:
  - might switch server you're on so save before you toggle

GPU Time
  - no guarantee of getting extra time, so be careful with how you use it
  - make sure to save results of long running tasks to disk as soon as the task ends (e.g. checkpoint your model params for deep learning networks) or you might lose the results of training after it disconnects
  - to keep your session alive during long running tasks, do the following:
  ```
  from workspace_utils import active_session

  with active_session():
      # do long-running work here
  ```

Submit Project
  - some workspaces can submit project directly just by clicking "Submit Project" button
  - if the button isn't there, you'll have to download your project files and submit (make sure all required files are included with proper file conversions)

Opening Terminal
  - click new menu button at top right of file browser view and select "Terminal"

# 4 - Updating to PyTorch v0.4
They've updated to PyTorch v0.4
  - all the code should work
  - we might see warnings that can be ignored

# 5 - Image Classifer Part 1: Development
Goal will be to implement an image classifier using PyTorch
  - refer to the rubric for guidance toward successful submission
  - Do not plagarize

The shape of your model can determine the model size when you need to save it for the checkpoint
  - avoid wide layers and use more hidden layers
  - you can open a terminal and enter `ls -lh` to see the size of the files
  - if checkpoint is great than 1GB, reduce the size of the classifier network and resave

# 6 - Image Classifier Part 1: Workspace
Goal is to train image classifier to recognize different species of flowers
  load and preprocess the image dataset
  - train the image classifier on your dataset
  - use the trained classifier to predict image content

# 7 - Image Classifier Part 2: Command Line App
Convert your trained deep neural network into an application others can use
  - pair of python scripts that run from command line
  - for testing, use the checkpoint you saved in the first part

Specifications
  - submission must include `train.py` and `predict.py`

train.py
  - train new network on dataset and save the model as a checkpoint
predict.py
  - uses trained network to predict the class for an input image

Suggestions
  - create as many files as you need
  - consider a file just for functions and classes relating to the model
  - another file for utility functions like loading data and preprocessing images
  - NOTE: Make sure to include all files in your final submission
  - use `argparse` for getting command line inut into the scripts

# 8 - Image Classifier Part 2: Workspace
Workspace for creating/submitting the project

# 9 - Rubric
See complete list of requirements [here](https://classroom.udacity.com/nanodegrees/nd089/parts/cacc6bbc-d42a-42d6-a488-bbc66f95e4c8/modules/1897532f-3a85-4e82-a26d-9f4e351260ed/lessons/c094a284-2f2a-482d-83d6-fbadf91630bd/concepts/b4162538-0215-4bd0-9357-5e1614028ccf)

# 10 - Project: Create Your Own Image Classifier 
This is where you go to submit.

To get credit, you must submit the following:
  - completed Jupyter Notebook from part 1 as HTML file
    - include any extra files you created that are necessary to run the code
  - the `train.py` and `predict.py` files from Part 2
    - include any extra files necessary to run those scripts

NOTE 1: You can download these files individually from the workspaces
NOTE 2: Do not include the data in the submission archive
