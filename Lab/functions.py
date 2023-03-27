import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler # importing StandardScaler

# using GridSearchCV to find the best parameters for the models

from sklearn.model_selection import GridSearchCV # importing GridSearchCV

# Importing chosen models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# importing the train_test_split function
from sklearn.model_selection import train_test_split

# importing the accuracy_score function
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, f1_score

# importing VotingClassifier
from sklearn.ensemble import VotingClassifier


# creating a function to split X and y of the dataset
def split_X_y(df, target):
    X = df.drop(target, axis=1)
    y = df[target]
    return X, y

# Train test split function
def train_test_split(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def train_and_test(model, X_train, X_test, y_train, y_test): # defining the function
    model.fit(X_train, y_train) # training the model
    y_pred = model.predict(X_test) # predicting the test set

# function to print the accuracy score, confusion matrix and classification report and plotting the confusion matrix
def print_metrics(model, y_test, y_pred):
    print("Accuracy score: ", accuracy_score(y_test, y_pred)) # printing the accuracy score
    print("Confusion matrix: ", confusion_matrix(y_test, y_pred)) # printing the confusion matrix
    print("Classification report: ", classification_report(y_test, y_pred)) # printing the classification report
    print("F1-score: ", f1_score(y_test, y_pred)) # printing the f1-score
    print("Model: ", model) # printing the model
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=["No Cardiovascular Disease", "Cardiovascular Disease"]).plot() # plotting the confusion matrix
    plt.show() # showing the plot



def find_best_params(model, params, X_train, y_train): 
    grid_search = GridSearchCV(model, params, cv=5, n_jobs=-1, verbose=2) # creating a GridSearchCV object, params is the parameters to try, cv is the number of folds, n_jobs is the number of jobs to run in parallel, verbose is the verbosity level
    grid_search.fit(X_train, y_train) # fitting the model
    print("Model", model) #printing the model
    print("Best parameters: ", grid_search.best_params_) # printing the best parameters
    print("Best score: ", grid_search.best_score_) # printing the best score
    return grid_search.best_params_ # returning the best parameters