# CSCE 5214 Project 2
# House Price Prediction Based on Location and Weather

This mini project builds a web application for house price prediction. The user would provide a range of desired house price, a range of desired temperature, and the month and year that he/she plans to buy a new house. The application then provides of list of state/city that have the house price and temperature meeting the user's need.

__Technologies:__
- Flask 0.12 or later
- Python 3.6
- numpy
- pandas
- sklearn
- HTML & JavaScript

__Installation Guide:__
- Create a virtual environment for this project
- Clone this repository and install the following Python's package: numpy, pandas, sklearn, matplotlib, and seaborn
- Run [data wrangling](data_wrangling.ipynb) Jupyter notebook
- Run [model](model.ipynb) Jupyter notebook
- Open Command prompt for the current environment and go to 'web_app' directory. Then, type the following commands:
  - pip install -r requirements.txt
  - python main.py
  - go to : http://0.0.0.0:5000/
  
 __Build and Deploy Web App:__
 ```
cd web_app
docker build -t housepricesapp .
docker push housepricesapp
docker run --rm -p 5000:5000 -it housepricesapp:latest
``` 

__If you want to skip the build process and just run the prebuilt Web App:__
```
docker run --rm -p 5000:5000 -it rmoreira/csce5214project2:latest
```