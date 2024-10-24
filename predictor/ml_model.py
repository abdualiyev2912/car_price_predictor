import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

def train_model():
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, 'CarPrice.csv')
    
    df = pd.read_csv(file_path)
    
    X = df[['enginesize', 'horsepower']] 
    y = df['price']
    
    numerical_features = ['enginesize', 'horsepower']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features)
        ])
    
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model_pipeline.fit(X_train, y_train)
    
    with open('car_price_model.pkl', 'wb') as f:
        pickle.dump(model_pipeline, f)
    
    print("Model trained and saved!")
    
    y_pred = model_pipeline.predict(X_test)
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

train_model()
