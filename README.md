# First-Principles Molecular Dynamics with Machine Learning

### Getting Started
 
1. Clone the repo and ```cd``` into its directory

1. Install Pip globally

  Follow the instuctions found [here](https://pip.pypa.io/en/stable/installing/).

1. Install Virtualenv

  ```pip install virtualenv```

1. Initiate virtual environment

  ```virtualenv venv```

1. Take ownership of start and install scripts

  ```chmod u+x scripts/*```

### Starting the virtualenv

  ```source ./scripts/run``` will start the virtualenv

  ```deactivate``` will terminate the session

### Installing dependences

  With the virtualenv running, do:

  ```source ./scripts/install```

### Adding dependencies

  You can add dependences with:

  ```pip install <package-name>```

  To save the new dependency, run:

  ```./scripts/save_reqs```

### Notes on how to set up on HPC

  1. ```wget "https://bootstrap.pypa.io/get-pip.py"```
  1. ```python --user get-pip.py```
  1. Add to bash profile
    ```
    export PATH=$PATH:~/.local/bin
    export PYTHONPATH=$PYTHONPATH:~
    ```
  1. Get running!
