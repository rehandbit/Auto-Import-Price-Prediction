# 1985 Auto Import Price Prediction

A Machine Learning web application that predicts automobile prices based on vehicle specifications, engine characteristics, and insurance risk ratings.


The project uses the **1985 Auto Imports dataset** and implements a complete machine learning workflow, from data ingestion and preprocessing to model training, evaluation, and deployment.


----

## Project Overview

The goal of this project is to build a regression model capable of predicting the price of an automobile based on its available features.

The Project includes:
* Data Ingestion and Preprocessing
* Handling of categorical and numerical features
* Feature transformation
* Training and evaluation of multiple regression models
* Model selection based on R² score
* Prediction pipeline for new input data
* Flask-based web application
* Deployment on Render

------


## Dataset
|Property|Details|
|--------|-------|
|Source|UCI Machine Learning Repository|
|Number of Records|201|
|Number of features|26|
|Target Variable|Price|
|Target Unit|USD|
|Problem Type|Regression|

------
## Project Structure
1985-auto-imports-price-prediction/
|
 src/



----
## Installation and Setup
1. Clone the Repository

    >git clone https://github.com/rehandbit/Auto-Import-Price-Prediction.git

2. Create a Virtual Environment

    >python -m venv venv

    Activate the virtual environment:

3.  Install Dependencies
    
    > pip install -r requirements.txt
------
## Train the Model
Run the training pipeline:

    >python src/pipeline/training_pipeline.py

The training process performs the required data processing, model training, evaluation, and artifact generation.

----
## Run the Application

    Start the Flask application:
    
    >python app.py

Open the local application in your browser:

    https://127.0.0.1:5000

-----
## Prediction
The web application allows users to provide automobile specifications through an input form

The application processes the submitted values using the same preprocessing pipeline used during model training and passes the transformed data to the trained model to generate a predicted automobile price.

-----

## Deployment
The application is deployed using Render

-----
## Model Evaluation
The primary evaluation metric used in this project is the R²(coefficient of determination) score.

The R² score measures how well the model explains the variation in automobile proces. A higher score indicates better performance on the evaluatied data.

The best reported result among the tested models was:
    
    Model: XGBoost <br>
    R² score: 0.948

----
## Future Improvements
Possible improvements to the project include:
* Hyperparameter tuning
* Cross-validation
* Additional regression models
* Feature importance analysis
* Model Monitoring
* Automated CI/CD deployement
* Improved error handling and input validation
