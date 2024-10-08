import numpy as np
from app.models.regression_model import RegressionModel
from app.utils.data_utils import read_file, preprocess_data, get_specific_df, determine_target
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class RegressionService:
    def __init__(self):
        self.model = RegressionModel()  # Your regression model (LinearRegression or custom model)
    
    def predict(self, X, y):
        X = np.array(X).reshape(-1, 1)
        y = np.array(y)
        self.model.fit(X, y)
        return self.model.predict(X)
    
    def predict_from_file(self, file):
        df = read_file(file)
        preprocessed = preprocess_data(df)
        res_ti = self.get_predicted_data(df=preprocessed, target='DU_TI', xes=['P_TI', 'IU_TI'])
        return res_ti
    
    def get_predicted_data(self, df, target, xes):
        # Split the data into features and target
        x, y = determine_target(df, target, xes)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Train the model on raw data (no scaling)
        self.model.fit(X_train, y_train)

        # Predict on the test set
        predictions = self.model.predict(X_test)
        rounded_predictions = [round(p) for p in predictions]

        feature_names = x.columns
        target_name = y.name
        coef = self.model.coef_
        intercept = self.model.intercept_

        # Merge the features and coef into a dictionary
        coef_list = dict(zip(feature_names, coef.tolist()))
        df.reset_index(inplace=True)
        df_json = df.to_json(orient='records')

        # Calculate performance metrics
        r2 = r2_score(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)

        return {
            'r2': r2,
            'mse': mse,
            'mae': mae,
            'predictions': rounded_predictions,
            'actual': y_test.tolist(),
            'features': feature_names.tolist(),
            'target': target_name,
            'coef_list': coef_list,
            'intercept': intercept,
            'df': df_json
        }
    
    def predict_raw(self, input_data):
        # Ensure input is in 2D shape for prediction
        input_data = np.array(input_data).reshape(1, -1)

        # Predict using the model with raw input data (no scaling)
        predicted_value = self.model.predict(input_data)
        
        return predicted_value
    
    def get_raw_budget(self, file):
        df = read_file(file, sheet_name='budgets')
        cleaned = self.clean_budget(df)
        json_df = cleaned.to_json(orient='records')
        return json_df
    
    def clean_budget(self, df):
        df_budget = df.copy()
        df_budget.rename(columns={'Tahun': 'Year'}, inplace=True)
        return df_budget
