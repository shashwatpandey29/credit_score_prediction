# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Load the dataset
file_path = 'data.csv'
data = pd.read_csv(file_path)

# Display the first few rows and column names of the dataset
print("First few rows of the dataset:")
print(data.head())
print("\nColumns of the dataset:")
print(data.columns)

# Clean the data (removing quotes and renaming if necessary)
data.columns = data.columns.str.replace("'", "").str.strip()
data = data.replace("'", "", regex=True)

# Check for missing values and data types
print("\nData info:")
print(data.info())

# Encode categorical variables
categorical_columns = data.select_dtypes(include=['object']).columns
le = LabelEncoder()

for column in categorical_columns:
    data[column] = le.fit_transform(data[column])

# Separate features (X) and target (y)
X = data.drop('good', axis=1)  # Assuming 'good' is the target column
y = data['good']

# Clean the feature names
X.columns = X.columns.str.replace(r'\[', '', regex=True)\
                     .str.replace(r'\]', '', regex=True)\
                     .str.replace(r'<', '', regex=True)\
                     .str.replace(r'>', '', regex=True)\
                     .str.strip()

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up the parameter grid for tuning with class weights and additional hyperparameters
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 9, 11],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1, 0.2],  # Additional parameter to control overfitting
}

# Initialize the XGBoost classifier with class weights
model = XGBClassifier(scale_pos_weight=(sum(y_train == 0) / sum(y_train == 1)), random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=model, 
                           param_grid=param_grid, 
                           scoring='f1',  # Use f1 score for imbalanced classes
                           cv=5, 
                           verbose=1, 
                           n_jobs=-1)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Best parameters and best estimator
print(f"\nBest Parameters: {grid_search.best_params_}")
best_model = grid_search.best_estimator_

# Make predictions on the test set with the best model
y_pred = best_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Output the results
print(f"\nAccuracy: {accuracy}")
print("\nClassification Report:")
print(report)
