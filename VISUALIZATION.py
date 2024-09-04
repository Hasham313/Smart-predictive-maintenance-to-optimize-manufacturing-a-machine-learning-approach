# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the preprocessed data
X_train = pd.read_csv('X_train_preprocessed.csv')
y_train = pd.read_csv('y_train_preprocessed.csv')

# Combine features and target for easier visualization
df_train = pd.concat([X_train, y_train], axis=1)

# Step 1: Visualize the Distribution of Each Feature

# Histograms
X_train.hist(figsize=(12, 10), bins=20)
plt.suptitle('Histograms of Numerical Features')
plt.show()

# Box plots to check for outliers
plt.figure(figsize=(12, 10))
sns.boxplot(data=X_train)
plt.title('Box Plots of Numerical Features')
plt.xticks(rotation=90)
plt.show()

# Step 2: Visualize Relationships Between Features and the Target Variable

# Pair plot to visualize pairwise relationships between features and the target variable
sns.pairplot(df_train)
plt.suptitle('Pair Plot of Features and Target', y=1.02)
plt.show()

# Step 3: Visualize the Correlation Matrix

# Calculate the correlation matrix
correlation_matrix = df_train.corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.show()
