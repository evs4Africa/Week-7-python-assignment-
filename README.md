import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target  # Adding the target column
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})  # Convert to species names

# Display first few rows
print(df.head())


# Check data types and missing values
print(df.info())

# Check for missing values
print(df.isnull().sum())


# Fill missing values with the column mean (if applicable)
df.fillna(df.mean(), inplace=True)


# Summary statistics
print(df.describe())


# Compute mean sepal length for each species
print(df.groupby('species')['sepal length (cm)'].mean())


Identify patterns
Setosa has the smallest sepal length on average.

Virginica has the largest petal length and petal width. 


![1000091792](https://github.com/user-attachments/assets/017104f1-00fd-42df-8b1a-7e0ea43e9690)
