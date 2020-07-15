import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.model_selection import  train_test_split
from sklearn import  model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pickle
from sklearn import preprocessing
from sklearn.externals import joblib

df = pd.read_csv("D:/pythoncodes/heat.csv")
y1_test = ""
rand_value = ""
rand_value_pred = ""
def heart_predict():
    #print(df.head())
    #print(df.shape)
    #print(df.size)
    #x is number of rows and y  is values of column
    #print(df.columns)
    col_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang', 'oldpeak', 'slope', 'ca', 'thal']
    x = df[features]
    y = df.target
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.30)
    rnd_fst = RandomForestClassifier()
    rnd_fst.fit(x_train,y_train)
    rnd_fst_pred = rnd_fst.predict(x_test)
    #print(rnd_fst_pred)
    print("accuracy of randomforest :" , metrics.accuracy_score(y_test,rnd_fst_pred))
    y1_test = y_test
    rand_value = rnd_fst
    rand_value_pred = rnd_fst_pred
    #print(type(rnd_fst))
    #save file name
    filename = "Final_trained_model.sav"
    pickle.dump(rnd_fst,open(filename,'wb'))

    # #load models
    # loaded_model = pickle.load(open(filename,'rb'))
    #
    # name = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca',
    #         'thal']
    # temp = []
    # for i in name:
    #     num = input(i)
    #     temp.append(num)
    #     # print(temp)
    # temp = np.array(temp).reshape((1, -1))
    # print(loaded_model.predict(temp))
    # # print(model.predict(temp))


heart_predict()
#save_model()