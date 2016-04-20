# Comparison of Gaussian, Arima & ANN in stock market prediction
---

Data set is taken from Yahoo! finance for MSFT. Gaussian Process is been tested for various kernel functions like absolute exponential, squared exponential etc.  

## Running Tests
        
        python gp.py
        python gpall.py
        Rscript arima_ann.R

## Files

- gpclose.py contains the source code for  gaussian process with considering closing price as features.
- gpall.py contains the source code for  gaussian process with considering all (open, high, low, close) as features.
- arima_ann.R contains the source code for arina and ann algorithms to do the prediction
- data.csv contains dataset for MSFT, taken from Yahoo! finance.

### Requirements
- [numpy](http://www.numpy.org/)
- pylab 
- [sklearn](https://pypi.python.org/pypi/scikit-learn/0.15.2)
    - [Cython](https://pypi.python.org/pypi/Cython/)
    - [SciPy Stack](http://www.scipy.org/install.html)

