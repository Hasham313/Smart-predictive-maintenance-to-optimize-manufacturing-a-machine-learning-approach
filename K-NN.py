import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE

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

# Drop rows with NaN values
df = df.dropna()

# Define features and target
X = df[['FM1', 'FM1_rolling_mean', 'FM1_lag_1', 'FM1_lag_2', 'hour', 'day']]
y = df['target']

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Initialize the K-NN model
knn = KNeighborsClassifier()

# Cross-validation with Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(knn, X_train, y_train, cv=skf, scoring='accuracy')

print("Cross-Validation Accuracy Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

# Fit the K-NN model on the training data
knn.fit(X_train, y_train)

# Predict on the test set
y_pred = knn.predict(X_test)

# Evaluate the model
print("Test Set Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Predict probabilities on the test set
y_probs = knn.predict_proba(X_test)[:, 1]

# Plot the ROC Curve to analyze the trade-off between sensitivity and specificity
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='K-NN (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()
