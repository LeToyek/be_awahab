import numpy as np
from app.models.regression_model import RegressionModel
from app.utils.data_utils import read_file, preprocess_data,get_specific_df,determine_target
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class RegressionService:
    def __init__(self):
        self.model = RegressionModel()
        self.scaler = StandardScaler()
    
    def predict(self, X, y):
        X = np.array(X).reshape(-1, 1)
        y = np.array(y)
        self.model.fit(X, y)
        return self.model.predict(X)
    
    
    def predict_from_file(self, file):
        df = read_file(file)
        
        preprocessed =  preprocess_data(df)
        
        df_ti = get_specific_df(preprocessed,'TI')
        df_si = get_specific_df(preprocessed,'SI')
        
        res_ti = self.get_predicted_data(df_ti,'Jml_Mhs_TI')
        res_si = self.get_predicted_data(df_si,'Jml_Mhs_SI')
        
        return res_ti,res_si
    
    def get_predicted_data(self,df,target):
        x,y = determine_target(df,target)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self.model.fit(X_train_scaled, y_train)
        predictions = self.model.predict(X_test_scaled)
        r2 = r2_score(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        return {
            'r2': r2,
            'mse': mse,
            'mae': mae,
            'predictions': predictions.tolist()
        }
    
        
        
