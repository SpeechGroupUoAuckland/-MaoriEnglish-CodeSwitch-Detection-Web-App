# MaoriEnglish-CodeSwitch-Detection-Web-App

*This project is based on <https://github.com/MaoriEnglish-Codeswitch/MaoriEnglish-CodeSwitch-Detection>*

## Environment Setup

1. Install Anaconda/Miniconda
2. Add conda to your path
3. Run script `envInstall.bat` to create the dev environment

## File Specification

- `tree.txt` - The file structure of the project with a depth of 3
- `envInstall.bat` - The script to create the dev environment
- evaluation folder - Contains the evaluation scripts and the results
- biLSTM folder - Contains the training and testing code for the biLSTM model
- mbert folder - Contains the training and testing code for the mbert model
- models folder - Contains the trained models
- web folder - Contains the frontend, backend and the standalone app as well as the models for the web app
- `dbAnalysis.ipynb` - The notebook to analyse the database
- `modelTest.py` - The script to test the models
- `semiAutoTestSetGen.py` - The script to use Maori word rules to generate the test set
- `size2LabelGen.py` - The script to generate the training set with a window size of 2
- `size3LabelGen.py` - The script to generate the training set with a window size of 3

## Usage

0. Run `semiAutoTestSetGen.py`, `size2LabelGen.py` and `size3LabelGen.py` to generate the training and testing sets
1. Run the Python scripts in the biLSTM and mbert folders to train the base models.
2. Copy and rename the trained models to the models folder in the web folder.
3. Run `runServers.bat` script in the web folder to start the backend and frontend servers.
4. Run `modelTest.py` to test the models.
5. Evaluate the models using the evaluation scripts in the evaluation folder.
