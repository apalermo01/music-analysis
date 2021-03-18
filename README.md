# Music-Analysis
## Semester project for ITCS 5156 (Applied Machine Learning)

## Installation / setup using anaconda

To set up virtual environment in anaconda: 

In the anaconda prompt, change to the directory of the fetched repo<br>
ex: for me, conda's base directory is C:\Users\<username>\, so I would type: <br>
`cd Documents/GitHub/music-analysis`<br>

Then create a virtual environment using the environment.yml file <br>
`conda env create -f environment.yml --name <environment name>`

If everything goes according to plan, you now have dependencies identical to what everyone else is currently using. <br>
If you add a new dependency: <br>
`conda env export > environment.yml` before comitting

If Github is having trouble rendering notebooks, use this: 
https://nbviewer.jupyter.org/



For more info on conda environments: <br>
https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html<br>
https://stackoverflow.com/questions/48787250/set-up-virtualenv-using-a-requirements-txt-generated-by-conda<br>
https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html<br>

# Instructions for running models / experiments


The run_model.py script acts as a pipeline to train and validate models using the training set of the nsynth dataset. This pulls data from from the zipped training set (in a .tar.gz file), pulls the files sequentially\* and returns the data in batchs.

Example:
If we set num_files = 100 and num_batches = 3, the script will extract and process files 1-100, train the model, then extract and process files 101-200, then do more training\*\*

The model itself is defined in sequential_model.py. run_model initializes the object defined by the sequential_model class. The reasoning for this setup is to allow for easy modification for each model so that we can easily keep track of various attributes and hyperparameters. 

When adding new models, I suggest we define all of them be seperate classes, this can either occur by making a dedicated file for each model (ideal if we're working with a huge number of different architectures) or by putting multiple classes in the same script (ideal if we're working with a few architectures and varying the hyperparameters for each one). 


\* **This is a known issue and needs to be fixed in an update. Taking this approach will probably lead to overfitting.**

\*\* I am pretty sure this is what's going on. Current tests with only a sequential model are not giving accuracies above ~0.25 and varying wildly. We need to make sure tha it's actually not re-training the model from scratch for each batch. This will be something to consider if future models that we try don't improve after the first batch. 
