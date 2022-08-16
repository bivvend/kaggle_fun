# evaluate lightgbm for classification
from lightgbm import LGBMClassifier
import pandas as pd

from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings("ignore")


TRAIN_DATA = "C:\\repos\\kaggle_fun\\Titanic\\ProcessedData\\train_data_added_features_1.csv"
TEST_DATA =  "C:\\repos\\kaggle_fun\\Titanic\\ProcessedData\\test_data_added_features_1.csv"

SUBMISSION_OUT = "C:\\repos\\kaggle_fun\\Titanic\\Results\\lgbm_submission_added_features_1.csv"

if __name__ == "__main__":


    #Load in the data
    raw_df = pd.read_csv(TRAIN_DATA)

    test_df = pd.read_csv(TEST_DATA)

    initial_test = test_df.copy(deep= True)
    #print(initial_test.head())

    X = raw_df.drop(["Survived","PassengerId"], axis=1)

    Y = raw_df["Survived"]

    # evaluate the model

    lgbm = LGBMClassifier(random_state=0)
    params = {
        "boosting_type": ["gbdt", "dart", "goss"],
        "learning_rate": [0.1, 0.05, 0.01],
        "n_estimators": [10, 50, 100, 300]
    }
    clf = GridSearchCV(lgbm, params, cv=10)
    clf.fit(X, Y)
    print("Best hyperparameter:", clf.best_params_)

    y_pred = clf.predict(X)
    print(classification_report(Y, y_pred))




    TestX = test_df.drop("PassengerId", axis=1)

    results = clf.predict(TestX)


    print(results)

    #Make a submission

    sub_df = initial_test
    sub_df["Survived"] = results

    sub_df = sub_df[["PassengerId", "Survived"]]

    print(sub_df.head())

    sub_df.to_csv(SUBMISSION_OUT, index=False)