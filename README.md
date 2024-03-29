# DSCI 510 - Final Project Template  

This repository is an example for the final project in the DSCI 510 class  
Included in this documentation are all of the necessary parts that grading is going to be based on  

# Dependencies

> A list of all of the dependencies used, included their version number  
```
matplotlib==3.6.2
numpy==1.23.3
pandas==1.5.0
plotly==5.11.0
requests==2.27.1
scipy==1.9.3
seaborn==0.12.1
statsmodels==0.13.5

```
# Installation

How to install the requirements necessary to run your project:  

```
pip install -r requirements.txt
```

# Running the project

Typically, a single file is called to run the project (something along the lines of)  

```
python final.py
```

# Methodology

> What kind of analyses or visualizations did you do? (Similar to Homework 2 Q3, but now you should answer based on your progress, rather than just your plan)  


# Visualization

- which visualization methods did we use
- why did we chose this particular way of visualizing data
- what insights are revealed through the means of this visualization

> [Bonus] Did you do any advanced visualizations? Briefly describe how you do it, and why it’s a great visualization  

# Future Work

Given more time, what is the direction that you would want to take this project in? 

Since outliers occur in January and December, I would make the data in these two months into a separate model while data in the rest of the year is into another different model. Current models need to fit into the January and December data better. The predictive values giving minimum temperature in January and December are much lower than the observed values. Separating them might help me research factors that relate to covid cases. I am also interested in possible reasons for the vast differences between these two models. 

What’s more, this project proves that there is a relationship between minimum temperature and covid cases. If I want to investigate further though not in a data science discipline, the next step will be to design an experiment to investigate their causal relationship for future disease prevention and decline in their transmission and break their stability. 

 
