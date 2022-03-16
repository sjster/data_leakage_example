

### What does it do?

This tests data leakage on three regression problems. Setting the 'PREPROCESS_BEFORE_SPLIT' flag allows you to perform preprocessingbefore the data is split.

The results are measure using R squared, MAE and MSE. The results are written out to CSV files and plotted using plotgraphs.py.

### How do you run it?

Run the following one level above


1. Set the PREPROCESS_BEFORE_SPLIT flag 

```
2. python source/dataleakage.py
```
 
```
3. python source/plotgraphs.py
```

### Documentation

Run the following one level above

```
pdoc source 
```

### Testing 

Run the the following pytest command one level above

```
pytest -v tests/unittests.py
```

For code coverage information use the package pytest-cov which can be invoked from pytest as

```
pytest --cov=dataleakage tests/unittests.py
```

For code coverage results in HTML format, including the lines of code that are not covered use the following

```
pytest --cov=dataleakage --cov-report=html tests/unittests.py
```
