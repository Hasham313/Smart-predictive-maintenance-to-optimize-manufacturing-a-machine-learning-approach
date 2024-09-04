# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Load the dataset
file_path = 'D:Downloads/iiot_30min_norm.csv'  # Update this path if necessary
df = pd.read_csv(file_path)

print("Missing Values per Column before Imputation:")
print(df.isnull().sum())

num_cols = df.select_dtypes(include=['float64', 'int64']).columns
imputer = SimpleImputer(strategy='mean')
df[num_cols] = imputer.fit_transform(df[num_cols])

cat_cols = df.select_dtypes(include=['object']).columns
imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = imputer.fit_transform(df[cat_cols])

# Step 2: Encode Categorical Variables

# Encode categorical variables using Label Encoding
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Step 3: Feature Scaling

# Scale numerical features using StandardScaler
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Step 4: Split the Data into Training and Testing Sets

# Assuming the last column is the target variable, adjust accordingly if different
X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]   # Target

# Split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display shapes of the resulting datasets
print("\nShape of Training Data:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print("\nShape of Testing Data:")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

# Optionally, save the preprocessed data to new CSV files
X_train.to_csv('X_train_preprocessed.csv', index=False)
X_test.to_csv('X_test_preprocessed.csv', index=False)
y_train.to_csv('y_train_preprocessed.csv', index=False)
y_test.to_csv('y_test_preprocessed.csv', index=False)
