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
