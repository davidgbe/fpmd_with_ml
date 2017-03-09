# First-Principles Molecular Dynamics with Machine Learning

### Getting Started
 
1. Clone the repo and ```cd``` into its directory

1. Install Pip globally

  Follow the instuctions found [here](https://pip.pypa.io/en/stable/installing/).

1. Initiate virtual environment

  ```python3 -m venv .```

1. Take ownership of start and install scripts

  ```chmod u+x user/bin/*```

### Starting the virtualenv

  ```source fpmd_run``` will start the virtualenv

  ```deactivate``` will terminate the session

### Installing dependences

  With the virtualenv running, do:

  ```fpmd_install```

### Adding dependencies

  You can add dependences with:

  ```pip install <package-name>```
  
  To save new dependencies, run:
  
  ```source fpmd_save_reqs```

### Notes on how to set up on HPC

  1. ```wget "https://bootstrap.pypa.io/get-pip.py"```
  1. ```python --user get-pip.py```
  1. Add to bash profile
    ```
    export PATH=$PATH:/usr/usc/python/default/bin
    export PATH=$PATH:~/fpmd_with_ml/user/bin
    export PATH=$PATH:~/lib/python3.5/site-packages
    export PYTHONPATH=$PYTHONPATH:~
    ```
  1. Get running!
