import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_diabetes, load_iris, load_boston, fetch_california_housing

PREPROCESS_BEFORE_SPLIT = False

def train_and_evaluate(fn, X_data_train, X_data_test, y_data_train, y_data_test):
    fn.fit(X_data_train, y_data_train)
    print(fn.score(X_data_test, y_data_test))
    y_data_predicted = fn.predict(X_data_test)
    print('R2 score is ',r2_score(y_data_test, y_data_predicted))
    print('Mean absolute error is ',mean_absolute_error(y_data_test, y_data_predicted))
    print('Mean squared error is ',mean_squared_error(y_data_test, y_data_predicted))

def get_diabetes_data():
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df['MEDV'] = diabetes.target
    X = df.loc[:, df.columns != 'MEDV']
    y = df.loc[:, df.columns == 'MEDV']
    return(X,y)

def get_boston_data():
    iris = load_boston()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['MEDV'] = iris.target
    X = df.loc[:, df.columns != 'MEDV']
    y = df.loc[:, df.columns == 'MEDV']
    return(X,y)

def get_california_data():
    california = fetch_california_housing()
    df = pd.DataFrame(california.data, columns=california.feature_names)
    df['MEDV'] = california.target
    X = df.loc[:, df.columns != 'MEDV']
    y = df.loc[:, df.columns == 'MEDV']
    return(X,y)

def standardize_data(X, y):
    X = (X - X.mean())/ X.std(axis=0)
    #y = (y - y.mean())/ y.std(axis=0)
    return(X, y)

def normalize_data(X, y):
    X = (X - X.min())/ (X.max() - X.min())
    #y = (y - y.mean())/ y.std(axis=0)
    return(X, y)

def run_pipeline(X, y, preprocess_fn, tag):
    print(f'====== {tag} ==== Preprocess before split is {PREPROCESS_BEFORE_SPLIT} ============{preprocess_fn} =============')

    if(PREPROCESS_BEFORE_SPLIT):
        if(preprocess_fn):
            X, y = preprocess_fn(X,y)

    seed = [1,2,3,10,100,200,500,1000]
    for elem in seed:
        X_data_train, X_data_test, y_data_train, y_data_test = train_test_split(X, y, test_size=0.3, random_state=elem)
        if(not PREPROCESS_BEFORE_SPLIT):
            if(preprocess_fn):
                X_data_train, y_data_train = preprocess_fn(X_data_train, y_data_train)
                X_data_test, y_data_test = preprocess_fn(X_data_test, y_data_test)

        lr = LinearRegression()
        train_and_evaluate(lr, X_data_train, X_data_test, y_data_train, y_data_test)

X, y = get_boston_data()
run_pipeline(X, y, standardize_data, 'BOSTON')
run_pipeline(X, y, normalize_data, 'BOSTON')
run_pipeline(X, y, None, 'BOSTON')

X, y = get_diabetes_data()
run_pipeline(X, y, standardize_data, 'DIABETES')
run_pipeline(X, y, normalize_data, 'DIABETES')
run_pipeline(X, y, None, 'DIABETES')

X, y = get_california_data()
run_pipeline(X, y, standardize_data, 'CALIFORNIA')
run_pipeline(X, y, normalize_data, 'CALIFORNIA')
run_pipeline(X, y, None, 'CALIFORNIA')
