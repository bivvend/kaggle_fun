
from pyexpat import model
import pandas as pd
# evaluate random forest algorithm for classification
from numpy import mean
from numpy import std
import tensorflow as tf
from tensorflow.keras import metrics, models, Input, Sequential 
from tensorflow.keras.layers import Dropout



TRAIN_DATA = "C:\\repos\\kaggle_fun\\Titanic\\ProcessedData\\train_data_simple2.csv"
TEST_DATA =  "C:\\repos\\kaggle_fun\\Titanic\\ProcessedData\\test_data_simple2.csv"

SUBMISSION_OUT = "C:\\repos\\kaggle_fun\\Titanic\\Results\\nn_submission_3.csv"

if __name__ == "__main__":
    

    #Load in the data
    raw_df = pd.read_csv(TRAIN_DATA)

    test_df = pd.read_csv(TEST_DATA)

    initial_test = test_df.copy(deep= True)
    #print(initial_test.head())

    X = raw_df.drop(["Survived","PassengerId"], axis=1)

    Y = raw_df["Survived"]

    #Build Kera sequential NN
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(8, input_shape=(9,)))
    model.add(tf.keras.layers.Dense(16))
    model.add(Dropout(0.5))
    model.add(tf.keras.layers.Dense(16))
    model.add(Dropout(0.5))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer='Adam', loss='mse')
    # This builds the model for the first time:
    model.fit(X, Y, batch_size=32, epochs=200)


    
    TestX = test_df.drop("PassengerId", axis=1)

    results = model.predict(TestX)

    #Make a submission

    sub_df = initial_test
    sub_df["Survived"] = results

    sub_df.loc[sub_df["Survived"] >= 0.5, ["Survived"]] = 1
    sub_df.loc[sub_df["Survived"] < 0.5, ["Survived"]] = 0


    sub_df = sub_df[["PassengerId", "Survived"]]

    sub_df = sub_df.astype({'Survived':'int32'})

    print(sub_df.dtypes)

    print(sub_df.head())

    sub_df.to_csv(SUBMISSION_OUT, index=False)