import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import KFold
from sklearn import model_selection, preprocessing, metrics
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":    
    print(os.listdir("../input"))
    train=pd.read_csv("../input/train.csv")
    test=pd.read_csv("../input/test.csv")
    print("Training dataset")
    print("----------------")
    train.info()
    train.head(5)
    print("Test dataset")
    print("----------------")
    test.info()
    test.head(5)