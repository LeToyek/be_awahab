from sklearn.linear_model import LinearRegression

class RegressionModel:
    def __init__(self):
        self.model = LinearRegression()
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    @property
    def coef_(self):
        return self.model.coef_
    
    @property
    def intercept_(self):
        return self.model.intercept_
    
    
