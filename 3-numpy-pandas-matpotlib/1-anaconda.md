# 1 - Instructor
Mat Leonard
  - PhD UC Berkley
Anconda is a package manager

# 2 - Introduction
Anaconda is distribution of libraries and software specifically for data science
  - comes with conda: package/environment manager
  - `conda list` shows you the packages that have been installed
  - `conda install <package>` installs additional packages 

# 3 - What is Anaconda?
Anaconda
  - distribution of software with `conda` and `python` and 150 packages
  - could just get `miniconda` if you just need `conda` and `python`
Managing Packages
  - `conda`s similar to `pip` except it's not for just python so you can isntall non-python packages
  - might still need `pip` for some packages that aren't available
Environments
  - allows you to separate and isolate the packages you're using for different projects
  - impossible to have 2 versions of numpy installed at once (instead make different environments)

# 4 - Installing Anaconda
It's available for [download](https://www.anaconda.com/download/)
Steps:
  - install it
  - run `conda upgrade conda` to upgrade conda itself
  - `conda upgrade --all` to get all the packages upgraded

# 5 - Managing Packages
Install with `conda install package_name`
  - can chain together with `package1 package2 package3`
  - specify an exact version with `conda install package_name=1.10`
  - dependencies are automatically installed
Uninstall with `conda remove package_name`
Update with `conda update package_name`
  - update all packages with `conda update --all`
List installed packages with `conda list`

# 6 - Managing Environments
Create an environment with `conda create -n env_name list of packages`
  - e.g. `conda create -n my_env numpy.`
  - specify version of python with `conda create -n py3 python=3`
Enter environment with `conda activate my_env`
When in an environment
  - env name will show up in the prompt `(my_env) ~ $`
  - now when you install packages they'll only be available in this environment
Leave environment with `source deactivate`

# 7 - More Environment Actions
Saving environments
  - useful to share envs with others
  - save packages to YAML with `conda env export > enrionment.yaml`
Loading environments
  - `conda env create -f environment.yaml`
Listing environments
  - `conda env list`
  - default environment used when you aren't in one is called `root`
Removing environments
  - `conda env remove -n env_name`

# 8 - Best Practices
Using Environments
  - have separate envs for Python 2 and Python 3 as a general playground for working in those languages
    - `conda create -n py2 python=2`
    - `conda create -n py2 python=2`
  - have separate environment for every project you're working on
Sharing Environments
  - when sharing code on GitHub, it's a good practice to make an environment file and include it in the repository
  - include a pip requirements.txt file using `pip freeze` for people not using conda

# 9 - On Python Versions at Udacity
ALways use Python 3
Most Python 2 breakages will be due to the print statement
  - `from __future__ import print_function` will allow you to use the print function in Python 2.6+ code
