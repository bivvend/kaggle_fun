import pandas as pd
import numpy as np
import sys

TRAIN_DATA = "D:\\repos\\kaggle_fun\\Titanic\\RawData\\train.csv"

if __name__ == "__main__":
    print("Titanic Data Importing")
    train_data_frame = pd.read_csv(TRAIN_DATA)
    print(train_data_frame.head())

    #Begin some data crunching
    #First plan - convert all columns into the range 0..1  -   May need some assumptions 
    #or additional data to help
    print("Scale class")
    min_class = train_data_frame["Pclass"].min()
    max_class = train_data_frame["Pclass"].max()

    train_data_frame["Pclass"] = 1.0 - (train_data_frame["Pclass"] - min_class) / (max_class - min_class)
    

    print("Convert Sex")
    train_data_frame.loc[ train_data_frame["Sex"]  == "male", ["Sex"]] = 0.0
    train_data_frame.loc[ train_data_frame["Sex"]  == "female", ["Sex"]] = 1.0
    print(train_data_frame.head(10))
    
    print("Convert Age")
    #Age has some NaN -  lets just replace these for the average for their class (class good age determiner...)
    
