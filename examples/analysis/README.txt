To run the analysis using Jupyter notebooks, you need to import the Utility Script in notebook_setup.py.
At the top of each notebook in the analysis folder, add the following lines:

# Import the setup function from notebook_setup.py
from notebook_setup import setup_notebook

# Call the setup function
setup_notebook()