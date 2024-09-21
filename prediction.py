import pickle
import pandas as pd

# Load the saved model
def load_model(filename="credit_model.pkl"):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

# Load the saved encoders
def load_encoders(filename="encoders.pkl"):
    with open(filename, 'rb') as file:
        encoders = pickle.load(file)
    return encoders

# Load the saved feature columns
def load_feature_columns(filename="feature_columns.pkl"):
    with open(filename, 'rb') as file:
        feature_columns = pickle.load(file)
    return feature_columns

# Function to make a prediction using the model and user input
def predict_credit_performance(model, encoders, feature_columns, user_input):
    # Convert user input into a DataFrame
    input_data = pd.DataFrame([user_input])

    # Encode categorical variables using the saved encoders
    categorical_columns = input_data.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        if column in encoders:
            input_data[column] = encoders[column].transform(input_data[column])
        else:
            raise ValueError(f"Unknown category column '{column}' in user input")

    # Reorder the input columns to match the training feature columns
    input_data = input_data.reindex(columns=feature_columns, fill_value=0)

    # Make the prediction
    prediction = model.predict(input_data)
    
    # Interpret the prediction
    result = "Good Performance" if prediction[0] == 1 else "Bad Performance"
    return result

# Main program to take user input and make a prediction
if __name__ == "__main__":
    # Load the saved model, encoders, and feature columns
    model = load_model()
    encoders = load_encoders()
    feature_columns = load_feature_columns()

    # Print encoded values for categorical columns
    for column, encoder in encoders.items():
        print(f"Encoded values for {column}: {dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}")

    # Example user input for bad performance using encoded values
    user_input = {
        'age': 22,
        'credit_amount': 50000,
        'credit_history': 0,  # Adjust based on your encoding
        'employment_status': 1,  # Adjust based on your encoding
        'housing': 1,  # Adjust based on your encoding
        # Add other features based on your dataset
    }

    # Make a prediction
    prediction = predict_credit_performance(model, encoders, feature_columns, user_input)

    # Output the result
    print(f"Prediction: {prediction}")
