Creating a comprehensive AI-powered optimizer for smart grid energy distribution involves several key elements, including data handling, a machine learning model for optimization, and evaluation components. Below is a simplified version of such a program using Python. This example assumes you have some grid data to work with and focuses on using scikit-learn for modeling.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AIPowerGridOptimizer:
    def __init__(self, data_path):
        """Initialize the optimizer with data."""
        self.data_path = data_path
        self.model = None

    def load_data(self):
        """Load data from a CSV file."""
        try:
            data = pd.read_csv(self.data_path)
            logging.info("Data loaded successfully.")
            return data
        except FileNotFoundError as e:
            logging.error(f"Data file not found: {e}")
            raise
        except pd.errors.EmptyDataError as e:
            logging.error(f"No data: {e}")
            raise
        except Exception as e:
            logging.error(f"An error occurred while loading the data: {e}")
            raise

    def preprocess_data(self, data):
        """Preprocess the data."""
        try:
            # Assume data has the following columns: "energy_demand", "supply" (target variable), "time_of_day", etc.
            features = data.drop('supply', axis=1)
            target = data['supply']

            # Normalize or process features as needed (simplified here)
            features_normalized = (features - features.mean()) / features.std()

            logging.info("Data preprocessing complete.")
            return features_normalized, target
        except KeyError as e:
            logging.error(f"Missing expected columns in data: {e}")
            raise
        except Exception as e:
            logging.error(f"Error during preprocessing: {e}")
            raise

    def train_model(self, X, y):
        """Train the machine learning model."""
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Initialize a Random Forest regressor
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)

            # Evaluate the model
            predictions = self.model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            logging.info(f"Model training complete with Mean Squared Error: {mse}")
        except Exception as e:
            logging.error(f"An error occurred during model training: {e}")
            raise

    def predict(self, new_data):
        """Make predictions with the trained model."""
        if not self.model:
            logging.error("Model is not trained yet.")
            raise Exception("Model not trained.")
        
        try:
            predictions = self.model.predict(new_data)
            logging.info(f"Predictions made successfully.")
            return predictions
        except Exception as e:
            logging.error(f"An error occurred during prediction: {e}")
            raise

if __name__ == "__main__":
    # Define path to your dataset here
    data_path = 'path_to_your_energy_data.csv'
    
    # Initialize the optimizer
    optimizer = AIPowerGridOptimizer(data_path)
    
    try:
        # Load and preprocess the data
        data = optimizer.load_data()
        features, target = optimizer.preprocess_data(data)
        
        # Train the model
        optimizer.train_model(features, target)
        
        # Example of making predictions
        # For demo purposes, we'll use a subset of the original features
        predictions = optimizer.predict(features.head())
        for i, prediction in enumerate(predictions):
            logging.info(f"Predicted supply for sample {i}: {prediction}")
    except Exception as e:
        logging.error(f"Failed to optimize power grid: {e}")
```

### Key Components:
- **Data Handling**: Loads and preprocesses the data. Assumes you have a CSV file with appropriate columns.
- **Model Training**: Utilizes a RandomForestRegressor model for example purposes. You can substitute this with more sophisticated models such as neural networks if needed.
- **Predictions and Error Handling**: Handles exceptions and uses logging to monitor progress and errors.
- **Error and Exception Management**: Catches potential errors during file operations, data preprocessing, and predictions.

In a real-world application, significant modifications and enhancements would be required, including but not limited to data validation, more complex feature engineering, robust scalability, and possibly integration with a larger energy management system.