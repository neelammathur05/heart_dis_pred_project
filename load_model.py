import pickle
import pandas as pd
import numpy as np
import joblib
from sklearn import metrics
import matplotlib.pyplot as plt

def user_input():
    lst = []
    age = int(input("enter your Age (Eg: 22) "))
    sex = int(input("enter your sex Female: 0, male:1(Eg: 0) "))
    cp = int(input("enter your cp (Eg: 3) "))
    trestbps = int(input("enter your trestbps (Eg: 145) "))
    chol = int(input("enter your chol (Eg: 360) "))
    fbs = int(input("enter your fbs  0 or 1 (Eg: 0) "))
    restecg = int(input("enter your restecg 0 or 1 (Eg: 1) "))
    thalach = int(input("enter your thalach (Eg: 150) "))
    exang = int(input("enter your exang 0 or 1 (Eg: 0) "))
    oldpeak = float(input("enter your oldpeak (Eg: 2.4) "))
    slope = int(input("enter your slope (Eg: 2) "))
    ca = int(input("enter your ca (Eg: 0) "))
    thal = int(input("enter your thal (Eg: 3) "))
    lst = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    #print(lst)
    return lst
def predict_result(a):
    filename = "C:/Users/user/PycharmProjects/python_projects_udemy/Final_trained_model.sav"
    loaded_model = pickle.load(open(filename, 'rb'))
    temp = np.array(lst_value).reshape((1, -1))
    #print()
    print(loaded_model.predict(temp))

if __name__ =="__main__":
    lst_value = user_input()
    predict_result(lst_value)














