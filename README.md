# titanic-disaster — Logistic Regression in Python & R (Dockerized)
Repository for homework3. Loads and processes the Titanic dataset programmatically.
Titanic Disaster — Logistic Regression in **Python** & **R**, and runs them inside **Docker containers** for reproducibility.

## Project Overview

This project applies logistic regression to predict the survival of passengers on the Titanic.
Both Python and R implementations follow similar steps:

Load and clean the dataset (train.csv, test.csv)

Handle missing values (Age, Embarked, Cabin)

Encode categorical variables

Train a logistic regression model

Save predictions to CSV files

All steps are containerized using Docker, so the project can be executed identically on any machine.

**Data**

train.csv – used to train the model

test.csv – used to generate predictions

Both files are located under:
src/data/

If missing, you can download them from Kaggle:
https://www.kaggle.com/c/titanic/data

## Running Instructions

Below are the steps to build and run both Docker environments.

### Python Version

1. Build the Docker image

```bash
docker build -t titanic-python -f Dockerfile .
```

2. Run the container

```bash
docker run --rm -v "$(pwd)/src/data:/app/src/data" titanic-python
```



3. Expected Output

Console will display:

Data summary and missing value report

Model accuracy

First 10 predictions on the test set

A new file will be generated:
src/data/predictions_py.csv

Example:

PassengerId,Survived
892,0
893,1
894,0
...

### R Version

**1. Build the Docker image**

```bash
cd src/titanic_r
docker build -t titanic-r .
```


**2. Run the container**

```bash
docker run --rm -v "$(pwd)/src/data:/app/data" titanic-r
```

**3. Expected Output**

Console output includes:

Model summary (coefficients, significance levels)

Training accuracy

First 10 predictions on the test set

A new file will be generated:
src/data/predictions_r.csv

Example:

PassengerId,Survived
892,0
893,1
894,0
...

## Implementation Details

### Python (src/titanic/main.py)

Loads data from src/data/train.csv

Replaces missing Age with the mean and drops the Cabin column

Encodes Sex (0 = male, 1 = female) and Embarked (C = 0, Q = 1, S = 2)

Builds a logistic regression model using scikit-learn

Evaluates training accuracy

Saves final predictions on test.csv → src/data/predictions_py.csv

### R (src/titanic_r/main.R)

Reads train.csv and performs data cleaning

Fills missing Age with the mean and missing Embarked with mode

Drops Cabin

Converts categorical variables (Sex, Embarked) into numeric codes

Fits a logistic regression model using glm()

Saves predictions → src/data/predictions_r.csv

## Dependencies

### Python

The required packages are listed in requirements.txt:
pandas
scikit-learn

### R

Packages installed automatically via install_packages.R:
install.packages(c("tidyverse", "dplyr"), repos = "https://cloud.r-project.org")

## Notes

Both environments are fully containerized — no local Python or R setup is required.

Prediction outputs (predictions_py.csv, predictions_r.csv) are ignored by .gitignore.

Model accuracy may vary slightly due to random initialization and rounding.

## Results Summary 

Python	Logistic Regression	predictions_py.csv	titanic-python	~0.78

R	Logistic Regression	predictions_r.csv	titanic-r	~0.80

**Author**: Wangchen He

Last Updated: October 31, 2025
