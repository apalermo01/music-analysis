#Music-Analysis
## Semester project for ITCS 5156 (Applied Machine Learning)

## Installation / setup using anaconda

To set up virtual environment in anaconda: 

In the anaconda prompt, change to the directory of the fetched repo
ex: for me, conda's base directory is C:\Users\<username>\, so I would type: 
`cd Documents/GitHub/music-analysis`

Then create a virtual environment using the environment.yml file 
`conda env create -f environment.yml --name <environment name>`

If everything goes according to plan, you now have dependencies identical to what everyone else is currently using. 
If you add a new dependency: 
`conda env export > environment.yml` before comitting
