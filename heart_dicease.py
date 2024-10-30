import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Specify the file path of the dataset
file_path = r'C:\Users\ioann\Downloads\heart.csv'
# Read the CSV file into a DataFrame
data = pd.read_csv(file_path)

# Display the first few rows of the data to understand its structure
print(data)

# Check for duplicate rows in the data
duplicates = data.duplicated()
# Print the number of duplicate rows found
print("Number of duplicate rows:", duplicates.sum())

# Remove duplicate rows from the DataFrame
data_cleaned = data.drop_duplicates()

# Display the cleaned DataFrame to verify duplicates are removed
print("Data after removing duplicates:")
print(data_cleaned)

# Calculate the correlation matrix to understand relationships between features
correlation_matrix = data.corr()

# Plot the correlation matrix using a heatmap for visualization
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Plot a countplot of the 'target' variable to see the distribution of heart disease incidence
plt.figure(figsize=(8, 6))
sns.countplot(x='target', data=data, palette='viridis')
plt.title('Distribution of Heart Disease Incidence')
plt.xlabel('Heart Disease (1 = Disease, 0 = No Disease)')
plt.ylabel('Count')
plt.xticks([0, 1], ['No Disease', 'Disease'])  # Custom labels for clarity
plt.show()

# Plot a countplot of the 'sex' variable to examine the gender distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='sex', data=data, palette='pastel')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.xticks([0, 1], ['Female', 'Male'])  # Custom labels for clarity
plt.show()

# Plot a countplot for the 'cp' (chest pain) column to understand the distribution of chest pain types
plt.figure(figsize=(8, 6))
sns.countplot(x='cp', data=data, palette='magma')
plt.title('Chest Pain Type Distribution')
plt.xlabel('Chest Pain Type')
plt.ylabel('Count')
plt.xticks([0, 1, 2, 3], ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'])  # Custom labels for each chest pain type
plt.show()

# Plot a countplot of 'fbs' (fasting blood sugar) categorized by 'target' to examine its relationship with heart disease
plt.figure(figsize=(8, 6))
sns.countplot(x='fbs', hue='target', data=data, palette='coolwarm')
plt.title('Fasting Blood Sugar Distribution by Heart Disease Incidence')
plt.xlabel('Fasting Blood Sugar (1 = >120 mg/dL, 0 = <=120 mg/dL)')
plt.ylabel('Count')
plt.legend(title='Heart Disease', labels=['No Disease', 'Disease'])  # Legend labels for clarity
plt.show()

# Plot a histogram of 'trestbps' (resting blood pressure) to observe its distribution
plt.figure(figsize=(8, 6))
sns.histplot(data['trestbps'], kde=True, color='skyblue')
plt.title('Distribution of Resting Blood Pressure')
plt.xlabel('Resting Blood Pressure (mm Hg)')
plt.ylabel('Frequency')
plt.show()

# Plot a boxplot comparing 'trestbps' by 'sex' to observe differences in resting blood pressure by gender
plt.figure(figsize=(8, 6))
sns.boxplot(x='sex', y='trestbps', data=data, palette='pastel')
plt.title('Resting Blood Pressure Comparison by Sex')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.ylabel('Resting Blood Pressure (mm Hg)')
plt.xticks([0, 1], ['Female', 'Male'])  # Custom labels for clarity
plt.show()

## Plot a histogram of 'chol' (serum cholesterol) to observe its distribution
plt.figure(figsize=(8, 6))
sns.histplot(data['chol'], kde=True)
plt.title('Distribution of Serum Cholesterol Levels')
plt.xlabel('Serum Cholesterol (mg/dL)')
plt.ylabel('Frequency')
plt.show()

# Define a list of continuous variables for further analysis
continuous_vars = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Loop through each continuous variable to plot its histogram and KDE (Kernel Density Estimate)
for var in continuous_vars:
    plt.figure(figsize=(8, 6))
    sns.histplot(data[var], kde=True)
    plt.title(f'Distribution of {var.capitalize()}')
    plt.xlabel(var.capitalize())
    plt.ylabel('Frequency')
    plt.show()
