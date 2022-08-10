import pandas as pd
import numpy as np
import sys
import os

OUT_PATH = "D:\\repos\\kaggle_fun\\Titanic\\ProcessedData"
TRAIN_DATA = "D:\\repos\\kaggle_fun\\Titanic\\RawData\\train.csv"
TEST_DATA = "D:\\repos\\kaggle_fun\\Titanic\\RawData\\test.csv"


def process_dataframe(df):
       
    print("Scale class")
    min_class = df["Pclass"].min()
    max_class = df["Pclass"].max()

    df["ScaledClass"] = 1.0 - (df["Pclass"] - min_class) / (max_class - min_class)
    

    print("Convert Sex")
    df.loc[ df["Sex"]  == "male", ["ScaledSex"]] = 0.0
    df.loc[ df["Sex"]  == "female", ["ScaledSex"]] = 1.0
    
    print("Convert Age")

    df["AgeRecorded"] = 1.0
    df.loc[df["Age"].isna(), ["AgeRecorded"]] = 0.0

    

    first_class_age_with_nan = df.loc[df["Pclass"] == 1, ["Age"]]
    first_class_age_no_nan = first_class_age_with_nan[first_class_age_with_nan["Age"].notna()]
    first_class_mean_age = first_class_age_no_nan["Age"].mean()
    print("Mean age of first class passengers = {0}".format(first_class_mean_age))

    second_class_age_with_nan = df.loc[df["Pclass"] == 2, ["Age"]]
    second_class_age_no_nan = second_class_age_with_nan[second_class_age_with_nan["Age"].notna()]
    second_class_mean_age = second_class_age_no_nan["Age"].mean()
    print("Mean age of second class passengers = {0}".format(second_class_mean_age))

    third_class_age_with_nan = df.loc[df["Pclass"] == 3, ["Age"]]
    third_class_age_no_nan = third_class_age_with_nan[third_class_age_with_nan["Age"].notna()]
    third_class_mean_age = third_class_age_no_nan["Age"].mean()
    print("Mean age of third class passengers = {0}".format(third_class_mean_age))

    #So lets first give all age NaNs their class specific average age
    df.loc[ (df["Pclass"] == 1) & (df["Age"].isna()), ["Age"]] = first_class_mean_age
    df.loc[ (df["Pclass"] == 2) & (df["Age"].isna()), ["Age"]] = second_class_mean_age
    df.loc[ (df["Pclass"] == 3) & (df["Age"].isna()), ["Age"]] = third_class_mean_age


    #Children are very likely to have survived so we should make sure people with  "Master/Miss etc" in their title are given a lower
    #Lets get the mean age of these people (not worrying about class)

    young_ones = df.loc[df["Name"].str.contains("master|miss", case=False) ]

    young_ones = young_ones.loc[young_ones["AgeRecorded"] == 0.0] 

    #If they are not married and have parents aboard are likely to be children so lets give them an arbirary age of 5
    young_ones = young_ones.loc[young_ones["Parch"] > 0] 

    df.loc[ (df["Name"].str.contains("master|miss", case=False)) & (df["AgeRecorded"] == 0.0) ,  ["Age"]] = 5

    #Check if there are any remaining NaN in Age column

    #print(df.loc[df["Age"].isna()].head())

    #So lets make a scaled age column
    df["ScaledAged"] = df["Age"] / df["Age"].max()

    #Check no NaNs in Fare
    df.loc[df["Fare"].isna(), ["Fare"]] = df["Fare"].mean()

    #Make the scaled fare column
    df["ScaledFare"] = df["Fare"] / df["Fare"].max()

    #Check no NaNs in SibSp and Parch
    df.loc[df["SibSp"].isna(), ["SibSp"]] = 0
    df.loc[df["Parch"].isna(), ["Parch"]] = 0

    #scale these also
    df["ScaledParch"] = df["Parch"] / df["Parch"].max()
    df["ScaledSibSp"] = df["SibSp"] / df["SibSp"].max()

    #print(df.head(50))

    #Need to do something sensible with cabins
    #First lets replace NaNs 
    df.loc[df["Cabin"].isna(), ["Cabin"]] = "XXX"


    #Then make a column for our cabin guesses

    #print(df.loc[df["Cabin"].str.contains("G")].head(50))
    cabin_dict = {"A":8.0, "B": 7.0, "C": 6.0, "D": 5.0, "E": 4.0, "F": 3.0, "G": 2.0, "T": 1.0}
    
    df["CabinGuess"] = "D"


    for key in cabin_dict:
        try:
            df.loc[df["Cabin"].str.contains(key, na = False), ["CabinGuess"]] = key
        except Exception as e:
            print(e)


    #If we don't have a cabin lets infer from class

    #A,B,C  first   D, E  Second   F, G Third    -> This is a simplification of the actual map, but it will do. 
    class_cabin_dict = {1: "B", 2: "E", 3: "G"}
    for key in class_cabin_dict:
        df.loc[ (df["Cabin"].str.contains("XXX", na = False)) & (df["Pclass"] == key), ["CabinGuess"]]  = class_cabin_dict[key]

    #Scale cabin values

    df["ScaledCabin"] = df.replace({"CabinGuess": cabin_dict})["CabinGuess"]
    df["ScaledCabin"] = df["ScaledCabin"]/df["ScaledCabin"].max()

    #Number of Cabins
    df["NumberOfCabins"] = df["Cabin"].apply(lambda x: len(str(x).split(" ")))
    df["ScaledNumberOfCabins"] = df["NumberOfCabins"] / df["NumberOfCabins"].max() 

    #Handle Embarked NaNs
    df.loc[df["Embarked"].isna(), ["Embarked"]] = "S" 
    #print(df.loc[df["Embarked"].isna()].head())
    embarked_dict = {"S": 0.3,  "C": 0.4,  "Q": 1.0}   #These are just mappings -  may be important as someone on the ship for longer may be able to escape more easily

    df["ScaleEmbarked"] = 0.3
    for key in embarked_dict:
        df.loc[ (df["Embarked"] == key), ["ScaledEmbarked"]]  = embarked_dict[key]
    df.loc[df["ScaledEmbarked"].isna(), ["ScaledEmbarked"]] = 0.3

    return df

if __name__ == "__main__":
    print("Titanic Train Data Importing")
    train_df = pd.read_csv(TRAIN_DATA)

    train_df = process_dataframe(train_df)
    print(train_df.head())

    base_train_df = train_df[["PassengerId", "Survived", "ScaledClass", "ScaledSex", "AgeRecorded", "ScaledFare", "ScaledParch", "ScaledSibSp", "ScaledCabin", "ScaledNumberOfCabins", "ScaledEmbarked"]]


    file_path = os.path.join(OUT_PATH, "train_data_simple2.csv")
    base_train_df.to_csv(file_path, index=False)

    print("Titanic Test Data Importing")
    test_df = pd.read_csv(TEST_DATA)

    print(test_df.head())

    test_df = process_dataframe(test_df)

    base_test_df = test_df[["PassengerId","ScaledClass", "ScaledSex", "AgeRecorded", "ScaledFare", "ScaledParch", "ScaledSibSp", "ScaledCabin", "ScaledNumberOfCabins", "ScaledEmbarked"]]

    print(base_test_df.columns[base_test_df.isna().any()].tolist())

    file_path = os.path.join(OUT_PATH, "test_data_simple2.csv")
    base_test_df.to_csv(file_path, index=False)





