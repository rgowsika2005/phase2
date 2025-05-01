# Patient Age Prediction Project
# Author: Gowsika.R

# 1. Import necessary libraries
# These appear to be the libraries used based on the document
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D

# 2. Data Loading
# The specific file name isn't mentioned in the document, so using a placeholder
# Replace 'your_dataset.csv' with the actual filename
data = pd.read_csv('your_dataset.csv')

# 3. Data Preprocessing
# Binary conversion of date_died feature
data['died'] = data['date_died'].apply(lambda x: 0 if pd.isna(x) else 1)

# Feature removal - dropping unnecessary columns
data = data.drop(['date_died'], axis=1)

# 4. Feature Engineering
# Patient type derivation
# Note: The specific columns used for patient type aren't explicitly mentioned
# This is a representative example based on the document
# You may need to adjust the columns based on your actual data

# Disease indicator features
# Creating features for patients with pre-existing conditions
# The exact code implementation isn't provided in the document
# This is a representative example:
# For each disease column, create a binary indicator
disease_columns = ['disease_1', 'disease_2', 'disease_3']  # Replace with actual column names
for column in disease_columns:
    data[f'{column}_present'] = data[column].apply(lambda x: 1 if x == 1 else 0)

# 5. Data Splitting
# Split data into training and testing sets (80/20 split)
X = data.drop(['age'], axis=1)  # Features - assuming 'age' is the target column
y = data['age']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Model Building
# The document doesn't specify which regression model was used
# Using a placeholder for LinearRegression as an example
from sklearn.linear_model import LinearRegression

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# 7. Model Evaluation
# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Square Error: {rmse}")

# 8. Visualization of Results

# Actual vs. Predicted Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r')
plt.xlabel('Actual Age')
plt.ylabel('Predicted Age')
plt.title('Actual vs Predicted Age')
plt.grid(True)
plt.show()

# 3D Contour Plot
# For a 3D visualization showing relationships between multiple variables
# This is a representative example - you'll need to select the appropriate variables
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Selecting two feature columns for demonstration
# Replace 'feature1' and 'feature2' with actual column names
feature1 = X_test.iloc[:, 0]  
feature2 = X_test.iloc[:, 1]

# Create the 3D scatter plot
scatter = ax.scatter(feature1, feature2, y_pred, c=y_test, cmap='viridis')

# Add labels and title
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Predicted Age')
ax.set_title('3D Visualization of Age Prediction')

# Add a color bar to show the mapping of colors to actual age values
plt.colorbar(scatter, ax=ax, label='Actual Age')

plt.tight_layout()
plt.show()
