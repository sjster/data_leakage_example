import sys
sys.path.append('./source/')
import numpy as np
import pytest
from dataleakage import *

def test_boston_data():
  X,y = get_boston_data()
  assert(np.shape(X) == (506, 13) and np.shape(y) == (506, 1))

def test_diabetes_data():
  X,y = get_diabetes_data()
  assert(np.shape(X) == (442, 10) and np.shape(y) == (442, 1))

def test_california_data():
  X,y = get_california_data()
  assert(np.shape(X) == (20640, 8) and np.shape(y) == (20640, 1))

@pytest.fixture(scope='module')
def get_boston():
    X, y = get_boston_data()
    yield(X,y)

@pytest.fixture(scope='module')
def get_california():
    X, y = get_california_data()
    yield(X,y)

@pytest.fixture(scope='module')
def get_diabetes():
    X, y = get_diabetes_data()
    yield(X,y)

def test_standardize_boston(get_boston):
    X, y = get_boston[0], get_boston[1]
    X_std, y_std = standardize_data(X, y)
    assert(np.allclose(X_std.mean(), 0.0) and np.allclose(X_std.std(), 1.0))

def test_standardize_california(get_california):
    X, y = get_california[0], get_california[1]
    X_std, y_std = standardize_data(X, y)
    assert(np.allclose(X_std.mean(), 0.0) and np.allclose(X_std.std(), 1.0))

def test_standardize_diabetes(get_diabetes):
    X, y = get_diabetes[0], get_diabetes[1]
    X_std, y_std = standardize_data(X, y)
    assert(np.allclose(X_std.mean(), 0.0) and np.allclose(X_std.std(), 1.0))

def test_normalize_boston(get_boston):
    X, y = get_boston[0], get_boston[1]
    X_std, y_std = normalize_data(X, y)
    assert(np.allclose(X_std.min(), 0.0) and np.allclose(X_std.max(), 1.0))

def test_normalize_california(get_california):
    X, y = get_california[0], get_california[1]
    X_std, y_std = normalize_data(X, y)
    assert(np.allclose(X_std.min(), 0.0) and np.allclose(X_std.max(), 1.0))

def test_normalize_diabetes(get_diabetes):
    X, y = get_diabetes[0], get_diabetes[1]
    X_std, y_std = normalize_data(X, y)
    assert(np.allclose(X_std.min(), 0.0) and np.allclose(X_std.max(), 1.0))

def test_run_pipeline_std(get_diabetes):
    X, y = get_diabetes[0], get_diabetes[1]
    r2, mae, mse = run_pipeline(X, y, standardize_data, 'DIABETES')
    assert(all(elem > 0 for elem in r2))
    assert(all(elem > 0 for elem in mae))
    assert(all(elem > 0 for elem in mse))

def test_run_pipeline_norm(get_diabetes):
    X, y = get_diabetes[0], get_diabetes[1]
    r2, mae, mse = run_pipeline(X, y, normalize_data, 'DIABETES')
    assert(all(elem > 0 for elem in r2))
    assert(all(elem > 0 for elem in mae))
    assert(all(elem > 0 for elem in mse))

def test_run_pipeline_none(get_diabetes):
    X, y = get_diabetes[0], get_diabetes[1]
    r2, mae, mse = run_pipeline(X, y, None, 'DIABETES')
    assert(all(elem > 0 for elem in r2))
    assert(all(elem > 0 for elem in mae))
    assert(all(elem > 0 for elem in mse))
