# 1 - Instructor
Jupyter Notebooks
  - convenient way to create & share document that contains text, code, videos, equations and images
Instructor
  - same as last lesson

# 2 - What are Jupyter Notebooks?
What are they?
  - web application that allows you to combine text, equations, code and visualizations in a shareable document
  - [example notebook](https://losc.ligo.org/s/events/GW150914/GW150914_tutorial.html)
  - they're rendered automaticall in GitHub
Literate Programming
  - proposed by Donald Knuth in 1984
  - documentation is written as a narrative alongside code instead of off on it's own
  - recently concept was extended to an entire programming language [eve](http://www.witheve.com/)
How it works
  - connect to server through browser
  - notebook is rendered as a web app
  - code you write in web app is sent through server to kernel
  - kernel runs code and sends it back to server
  - output is rendered back in the borwser
  - when you save it's written as a JSON file with a `.ipynb` file extension
  * note: notebooks are language agnostic and work with [these languages](https://github.com/jupyter/jupyter/wiki/Jupyter-kernels)


# 3 - Installing Jupyter Notebooks
Install with `conda install jupyter notebook`

You can also use pip `pip install jupyter notebook`

# 4 - Launching the Notebook Server
Launch Server with `jupyter notebook` in terminal/console
  - starts the server in the directory you ran the command in
  - notebook files will be saved in that directory
  - you'll generally start the server in the directory where your notebooks live
When command is run:
  - server home opens in browser (usually localhost:8888)
  - click on `new` on the right to make a new notebook, text file, folder or terminal
    - this will allow you to select the kernel
Tabs at the top:
  - files: shows all files/folders in current directory
  - running: list all currently running notebooks
  - clusters: no longer serves much purpose
  - conda: manage environments from within jupyter (if nb_conda is installed)

Install Notebook Conda to help manage environments
  - `conda install nb_conda`

Shutting Down
  - mark checkbox next to notebook on server home and click "Shutdown"
  - make sure to save first
  - can shut down all notebooks at once with `control + C` twice in the terminal

# 5 - Notebook Interface
Cell
  - Little box outlined in green
  - this is where you write/run your code
  - can be changed to render Markdown
  - the little play button runs the cell
  - up/down arrows move the cell
Play
  - runs code in the cell
  - running code in markdown mode renders markdown as text
Tool Bar
  - save (floppy disc)
  - + creates new cell
  - cut, copy, paste cells
  - run, stop, restart kernel
  - cell type (code, markdown, raw text, header)
  - command palette (see below)
  - cell toolbar (options for cells like using them as slides)
Command Palette
  - gives you search bar where you can search for various commands
Click on title to rename notebook
Save
  - auto saves occasionally
  - click the save icon
  - `esc` then `s`
Download
  - allows you to download the notebook in various formats


# 6 - Code Cells
This is where you'll do most of your work
  - any code executed in one cell is available in others

# 7 - Markdown Cells
Running a markdown cell renders it to formatted text
  - code blocks with three backticks
  - Start math mode 
    - wrap LaTeX in dollar signs `$y = mx + b$`
    - do a block with double `$$`

# 8 - Keyboard Shortcuts
You'll be faster at working with notebooks if you learn keyboard shortcuts
  - There's a demo notebook you can play around with (in our notebooks directory)

# 9 - Magic Keywords
Magic Commands
  - Special commands you run in cells that let you control the notebook itself
    - can perform system calls such as changing directors
    - e.g. set up matplotlib to work interactively in the notebook with `%matplotlib`
  - always preceded with one or two percent signs (for lines or blocks)
  * NOTE: Specific to Python kernel and may not work in other kernels
Timing Code
  - use `timeit` magic command to time how long it takes for a function to run
  - e.g. `%timeit my_method(25)`
  - time an intire cell with `%%timeit` at the top of the cell
Embedding Visualizations
  - useful when using matplotlib
  - use %matplotlib to set up matplotlib for interactive use in the notebook
  - `%matplotlib inline` will render the images in the notebook instead of somewhere else
    - NOTE: If image is fuzzy on higher res screen include `%config InlineBackend.figure_format = 'retina'` after inline command
Debugging In the Notebook
  - turn on interactive debugger using `%pdb`
  - if an error occurs, you can inspect variables in the current namespace


# 10 - Converting Notebooks
Notebooks are big `json` files with extension `.ipynb`
  - easy to convert to other formats
  - jupyter comes with `nbconvert` for converting to HTML, Markdown, Slideshows, etc
  - in terminal `jupyter nbconvert --to html notebook.ipynb`
Notes about types:
  - html is useful for sharing notebooks with people who aren't using notebooks
  - markdown is great for including a notebook in a blog/text editor

# 11 - Creating a Slideshow
Slideshows are created in notebooks like normal
  - you'll need to designate which cells are slides (and slide type)
  - View > Cell > Toolbar > Slideshow opens up the slide menu on  each cell
Types:
  - slides: full slides that you move through left to right
  - sub-slides: show up in slide show by pressing up or down
  - fragments: hidden at first then appear with a button press
  - skip: allows you to skip the cell in the slideshow
  - notes: leaves the cell as speaker notes
Running the slide show
  - `jupyter nbconvert notebook.ipynb --to slides` to convert to the necessary files for slideshow
    - you'l need to serve it with an http server to see the presentation
  - create and immediately see presentation with `jupyter nbconvert notebook.ipynb --to slides --post serve`

# 12 - Finishing Up
Learning these tools will increase productivity
  - one day, you will ask, "Why did it ever not be this way?"

