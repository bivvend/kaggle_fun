from sklearn.ensemble import RandomForestClassifier
import pandas as pd
# evaluate random forest algorithm for classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

TRAIN_DATA = "D:\\repos\\kaggle_fun\\Titanic\\ProcessedData\\train_data_simple2.csv"
TEST_DATA =  "D:\\repos\\kaggle_fun\\Titanic\\ProcessedData\\test_data_simple2.csv"

SUBMISSION_OUT = "D:\\repos\\kaggle_fun\\Titanic\\Results\\rfc_submission_1.csv"

if __name__ == "__main__":
    rfc_model = RandomForestClassifier(n_estimators= 100)

    #Load in the data
    raw_df = pd.read_csv(TRAIN_DATA)

    test_df = pd.read_csv(TEST_DATA)

    initial_test = test_df.copy(deep= True)
    #print(initial_test.head())

    X = raw_df.drop(["Survived","PassengerId"], axis=1)

    Y = raw_df["Survived"]

    # evaluate the model
    cv = RepeatedStratifiedKFold(n_splits=40, n_repeats=10, random_state=1)
    n_scores = cross_val_score(rfc_model, X, Y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    # report performance
    print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
    rfc_model.fit(X,Y)

    TestX = test_df.drop("PassengerId", axis=1)

    results = rfc_model.predict(TestX)
    print(results)

    #Make a submission

    sub_df = initial_test
    sub_df["Survived"] = results

    sub_df = sub_df[["PassengerId", "Survived"]]

    print(sub_df.head())

    sub_df.to_csv(SUBMISSION_OUT, index=False)