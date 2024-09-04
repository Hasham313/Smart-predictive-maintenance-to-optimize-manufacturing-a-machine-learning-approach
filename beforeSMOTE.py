import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer

# Load the dataset
file_path = 'D:Downloads/iiot_30min_norm.csv'
df = pd.read_csv(file_path)

# Feature Engineering: Create lag features and rolling statistics
df['FM1_rolling_mean'] = df['FM1'].rolling(window=3).mean()
df['FM1_lag_1'] = df['FM1'].shift(1)
df['FM1_lag_2'] = df['FM1'].shift(2)
df['hour'] = pd.to_datetime(df['TIME']).dt.hour
df['day'] = pd.to_datetime(df['TIME']).dt.day

# Create the target variable with a refined threshold
failure_threshold = df['FM1'].quantile(0.05)
df['target'] = (df['FM1'].rolling(window=5).min() < failure_threshold).astype(int)
df['target'] = df['target'].shift(-5).fillna(0).astype(int)

# Drop rows with NaN values or impute missing values
df = df.dropna()

# Define features and target
X = df[['FM1', 'FM1_rolling_mean', 'FM1_lag_1', 'FM1_lag_2', 'hour', 'day']]
y = df['target']

# Check for any remaining NaN values in X
print("Checking for NaN values in features before imputation:")
print(X.isnull().sum())

# Impute missing values (if any remain)
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Ensure no NaN values remain
print("Checking for NaN values after imputation:")
print(pd.DataFrame(X).isnull().sum().sum())

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, stratify=None)

# Train an SVM with class weights to address imbalance
model = SVC(kernel='rbf', class_weight='balanced', C=0.5, random_state=42)

# Stratified K-Fold Cross Validation to assess model performance
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')

print("Cross-Validation Accuracy Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Test Set Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
