import sys
sys.path.append('./source/')
import numpy as np
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
