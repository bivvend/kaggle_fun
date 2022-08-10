import pandas as pd
import numpy as np
import sys
import os

OUT_PATH = "D:\\repos\\kaggle_fun\\Titanic\\ProcessedData"
TRAIN_DATA = "D:\\repos\\kaggle_fun\\Titanic\\RawData\\train.csv"

if __name__ == "__main__":
    print("Titanic Data Importing")
    train_data_frame = pd.read_csv(TRAIN_DATA)
    

    #Begin some data crunching
    #First plan - convert all columns into the range 0..1  -   May need some assumptions 
    #or additional data to help
    print("Scale class")
    min_class = train_data_frame["Pclass"].min()
    max_class = train_data_frame["Pclass"].max()

    train_data_frame["ScaledClass"] = 1.0 - (train_data_frame["Pclass"] - min_class) / (max_class - min_class)
    

    print("Convert Sex")
    train_data_frame.loc[ train_data_frame["Sex"]  == "male", ["ScaledSex"]] = 0.0
    train_data_frame.loc[ train_data_frame["Sex"]  == "female", ["ScaledSex"]] = 1.0
    
    print("Convert Age")
    #Age has some NaN -  lets just replace these for the average for their class (class good age determiner...)
    #Could also use the "Master, Mr, Miss. Mrs" to roughly estimate age.  Masters more likely to be young etc.

    #Lets look to see if the lack of an age tells us anything first. 
    print("Mean of survived column = {0}".format(train_data_frame["Survived"].mean()))
    no_age_only = train_data_frame[train_data_frame["Age"].isna()]
    print("Mean of survived column for Age is NaN = {0}".format(no_age_only["Survived"].mean()))

    #Mean of survived column = 0.3838383838383838
    #Mean of survived column for Age is NaN = 0.2937853107344633
    #So not having an age stored is a learnable feature - likely poor records for those who died.
    #So will add a new column "AgeRecorded"
    train_data_frame["AgeRecorded"] = 1.0
    train_data_frame.loc[train_data_frame["Age"].isna(), ["AgeRecorded"]] = 0.0

    

    first_class_age_with_nan = train_data_frame.loc[train_data_frame["Pclass"] == 1, ["Age"]]
    first_class_age_no_nan = first_class_age_with_nan[first_class_age_with_nan["Age"].notna()]
    first_class_mean_age = first_class_age_no_nan["Age"].mean()
    print("Mean age of first class passengers = {0}".format(first_class_mean_age))

    second_class_age_with_nan = train_data_frame.loc[train_data_frame["Pclass"] == 2, ["Age"]]
    second_class_age_no_nan = second_class_age_with_nan[second_class_age_with_nan["Age"].notna()]
    second_class_mean_age = second_class_age_no_nan["Age"].mean()
    print("Mean age of second class passengers = {0}".format(second_class_mean_age))

    third_class_age_with_nan = train_data_frame.loc[train_data_frame["Pclass"] == 3, ["Age"]]
    third_class_age_no_nan = third_class_age_with_nan[third_class_age_with_nan["Age"].notna()]
    third_class_mean_age = third_class_age_no_nan["Age"].mean()
    print("Mean age of third class passengers = {0}".format(third_class_mean_age))

#   Mean age of first class passengers = 38.233440860215055
#   Mean age of second class passengers = 29.87763005780347
#   Mean age of third class passengers = 25.14061971830986

    #So lets first give all age NaNs their class specific average age
    train_data_frame.loc[ (train_data_frame["Pclass"] == 1) & (train_data_frame["Age"].isna()), ["Age"]] = first_class_mean_age
    train_data_frame.loc[ (train_data_frame["Pclass"] == 2) & (train_data_frame["Age"].isna()), ["Age"]] = second_class_mean_age
    train_data_frame.loc[ (train_data_frame["Pclass"] == 3) & (train_data_frame["Age"].isna()), ["Age"]] = third_class_mean_age


    #Children are very likely to have survived so we should make sure people with  "Master/Miss etc" in their title are given a lower
    #Lets get the mean age of these people (not worrying about class)

    young_ones = train_data_frame.loc[train_data_frame["Name"].str.contains("master|miss", case=False) ]

    young_ones = young_ones.loc[young_ones["AgeRecorded"] == 0.0] 

    #If they are not married and have parents aboard are likely to be children so lets give them an arbirary age of 5
    young_ones = young_ones.loc[young_ones["Parch"] > 0] 

    train_data_frame.loc[ (train_data_frame["Name"].str.contains("master|miss", case=False)) & (train_data_frame["AgeRecorded"] == 0.0) ,  ["Age"]] = 5

    #Check if there are any remaining NaN in Age column

    #print(train_data_frame.loc[train_data_frame["Age"].isna()].head())

    #So lets make a scaled age column
    train_data_frame["ScaledAged"] = train_data_frame["Age"] / train_data_frame["Age"].max()

    #Check no NaNs in Fare
    #print(train_data_frame.loc[train_data_frame["Fare"].isna()].head())

    #Make the scaled fare column
    train_data_frame["ScaledFare"] = train_data_frame["Fare"] / train_data_frame["Fare"].max()

    #Check no NaNs in SibSp and Parch
    #print(train_data_frame.loc[train_data_frame["SibSp"].isna()].head())
    #print(train_data_frame.loc[train_data_frame["Parch"].isna()].head())

    #scale these also
    train_data_frame["ScaledParch"] = train_data_frame["Parch"] / train_data_frame["Parch"].max()
    train_data_frame["ScaledSibSp"] = train_data_frame["SibSp"] / train_data_frame["SibSp"].max()

    #print(train_data_frame.head(50))

    #Need to do something sensible with cabins
    #First lets replace NaNs 
    train_data_frame.loc[train_data_frame["Cabin"].isna(), ["Cabin"]] = "XXX"

    #Then make a column for our cabin guesses

    #print(train_data_frame.loc[train_data_frame["Cabin"].str.contains("G")].head(50))
    cabin_dict = {"A":8.0, "B": 7.0, "C": 6.0, "D": 5.0, "E": 4.0, "F": 3.0, "G": 2.0, "T": 1.0}
    
    train_data_frame["CabinGuess"] = "D"

    for key in cabin_dict:
        train_data_frame.loc[train_data_frame["Cabin"].str.contains(key), ["CabinGuess"]] = key

    #If we don't have a cabin lets infer from class

    #A,B,C  first   D, E  Second   F, G Third    -> This is a simplification of the actual map, but it will do. 
    class_cabin_dict = {1: "B", 2: "E", 3: "G"}
    for key in class_cabin_dict:
        train_data_frame.loc[ (train_data_frame["Cabin"].str.contains("XXX")) & (train_data_frame["Pclass"] == key), ["CabinGuess"]]  = class_cabin_dict[key]

    #Scale cabin values

    train_data_frame["ScaledCabin"] = train_data_frame.replace({"CabinGuess": cabin_dict})["CabinGuess"]
    train_data_frame["ScaledCabin"] = train_data_frame["ScaledCabin"]/train_data_frame["ScaledCabin"].max()

    #Number of Cabins
    train_data_frame["NumberOfCabins"] = train_data_frame["Cabin"].apply(lambda x: len(x.split(" ")))
    train_data_frame["ScaledNumberOfCabins"] = train_data_frame["NumberOfCabins"] / train_data_frame["NumberOfCabins"].max() 

    #Handle Embarked NaNs
    train_data_frame.loc[train_data_frame["Embarked"].isna(), ["Embarked"]] = "S" 
    #print(train_data_frame.loc[train_data_frame["Embarked"].isna()].head())
    embarked_dict = {"S": 0.3,  "C": 0.4,  "Q": 1.0}   #These are just mappings -  may be important as someone on the ship for longer may be able to escape more easily

    for key in embarked_dict:
        train_data_frame.loc[ (train_data_frame["Embarked"] == key), ["ScaledEmbarked"]]  = embarked_dict[key]

    

    #Enough for a basic data set

    base_train_df = train_data_frame[["Survived", "ScaledClass", "ScaledSex", "AgeRecorded", "ScaledFare", "ScaledParch", "ScaledSibSp", "ScaledCabin", "ScaledNumberOfCabins", "ScaledEmbarked"]]
    print(base_train_df.head(50))

    file_path = os.path.join(OUT_PATH, "train_data_simple.csv")
    base_train_df.to_csv(file_path, index=False)


