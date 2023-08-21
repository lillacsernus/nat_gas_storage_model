
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from datetime import datetime
from datetime import date





def fit_quad(prices, x_len, xx = 48):
    months = np.arange(x_len)
    ones = np.ones(np.shape(months))
    X = np.column_stack((ones, months, months**2))
    x48 = np.column_stack((np.ones(np.shape(np.arange(xx))), np.arange(xx), (np.arange(xx))**2))
    x48T = x48.T
    beta = np.linalg.inv(x48T.dot(x48)).dot(x48T.dot(prices))
    return X.dot(beta)

def fit_sc(prices, w, x_len):
    months = np.arange(x_len)
    XT = np.vstack((np.sin(w*months), np.cos(w*months)))
    X = np.transpose(XT)

    #Calculate beta
    xT = np.vstack((np.sin(w*np.arange(48)), np.cos(w*np.arange(48))))
    x = np.transpose(xT)
    beta = np.linalg.inv(xT.dot(x)).dot(XT).dot(prices)
    return X.dot(beta)

def fit_lin(prices, x_len):
    months = np.arange(x_len)
    ones = np.ones(np.shape(months))
    X = np.column_stack((ones, months))
    #Calculate beta
    x48 = np.column_stack((np.ones(np.shape(np.arange(48))), np.arange(48)))
    x48T = x48.T
    beta = np.linalg.inv(x48T.dot(x48)).dot(x48T.dot(prices))
    return X.dot(beta)

def fit_combined(fit_1, fit_2):
    return fit_1 + fit_2




