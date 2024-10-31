import numpy as np  
import pandas as pd 
from sklearn.model_selection import GridSearchCV 
import random 
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
from sklearn.tree import DecisionTreeRegressor  
from sklearn.ensemble import RandomForestRegressor  
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score    
import joblib  # 用于保存和加载模型  
from pathlib import Path
import sqlite3
file_path = Path(__file__).resolve().parent.parent.parent 
  
class HousePricePredictor:  
    def __init__(self, data):  
        self.data = data  
        self.X = data.drop(['总价','单价'], axis=1)  
        self.y = data['总价']  
        self.models = {} 
        self.model_paths = {} 
  
    def split_data(self, test_size=0.1, random_state=42):  
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(  
            self.X, self.y, test_size=test_size, random_state=random_state)  
  
    def train_linear_regression(self):  
        model = LinearRegression()  
        model.fit(self.X_train, self.y_train)  
        self.models['linear_regression'] = model  
  
    def train_decision_tree_regressor(self):  
        model = DecisionTreeRegressor(random_state=42)  
        model.fit(self.X_train, self.y_train)  
        self.models['decision_tree_regressor'] = model
  
    def train_random_forest_regressor(self, n_estimators=100):  
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)  
        model.fit(self.X_train, self.y_train)  
        self.models['random_forest_regressor'] = model  
  
    def train_xgboost(self):  
        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)  
        model.fit(self.X_train, self.y_train)  
        self.models['xgboost'] = model  
  
    def predict(self, model_name):  
        if model_name in self.models:  
            return self.models[model_name].predict(self.X_test)  
        else:  
            raise ValueError("Model not found")  
  
    def evaluate(self, model_name):  
        y_pred = self.predict(model_name)  
        mse = mean_squared_error(self.y_test, y_pred)  
        rmse = np.sqrt(mse)  # 计算均方根误差  
        mae = mean_absolute_error(self.y_test, y_pred)  # 计算平均绝对误差  
        r2 = r2_score(self.y_test, y_pred)  # 计算决定系数  
        return mse, rmse, mae, r2
    
    @staticmethod  
    def print_evaluation_results(model_name, evaluator):  
        mse, rmse, mae, r2 = evaluator.evaluate(model_name)  
        print(f"{model_name} MSE: {mse}")  
        print(f"{model_name} RMSE: {rmse}")  
        print(f"{model_name} MAE: {mae}")  
        print(f"{model_name} R²: {r2}")  
  
    def save_model(self, model_name, model, path='models/'):  
        """  
        保存模型到指定路径  
        """
        model_path = f"{path}{model_name}.joblib"  
        joblib.dump(model, model_path)  
        self.model_paths[model_name] = model_path  
        print(f"Model {model_name} saved to {model_path}")  
  
    def load_model(self, model_name):  
        """  
        从指定路径加载模型  
        """  
        if model_name in self.model_paths:  
            model_path = self.model_paths[model_name]  
            model = joblib.load(model_path)  
            return model  
        else:  
            raise ValueError("Model not found")  
  
    def predict_with_saved_model(self, model_name, X):  
        """  
        使用已保存的模型进行预测  
        """  
        model = self.load_model(model_name)  
        return model.predict(X)   
    
if __name__ == "__main__":
    conn = sqlite3.connect(str(file_path) + '\\data\\processed\\' + 'GisProject.db')  
    query = "SELECT * FROM house_label;"  
    house_data = pd.read_sql_query(query, conn)  

    predictor = HousePricePredictor(house_data)  
    predictor.split_data()  
    # predictor.train_linear_regression()
    # print("Linear Regression MSE:", predictor.evaluate('linear_regression'))  
    # predictor.save_model('linear_regression', predictor.models['linear_regression']) 

    # predictor.train_decision_tree_regressor() 
    # print("Decision Tree Regressor MSE:", predictor.evaluate('decision_tree_regressor')) 
    # predictor.save_model('decision_tree_regressor', predictor.models['decision_tree_regressor'])

    # predictor.train_random_forest_regressor()  
    # print("Random Forest Regressor MSE:", predictor.evaluate('random_forest_regressor'))   
    # predictor.save_model('random_forest_regressor', predictor.models['random_forest_regressor'])   
    
    # predictor.train_xgboost() 
    # print("XGBoost MSE:", predictor.evaluate('xgboost'))  
    # predictor.save_model('xgboost', predictor.models['xgboost'])



    # 使用已保存的模型进行预测  
    # X_new = pd.DataFrame(...)  # 新的待预测数据  
    # predictions = predictor.predict_with_saved_model('xgboost', X_new)