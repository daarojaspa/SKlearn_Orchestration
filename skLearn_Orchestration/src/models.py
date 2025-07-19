import pandas as pd
import numpy as np 
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from utils.utils import Utils
class Models():
    def __init__(self):
        self.reggresors={
            'SVR':SVR(),
            'GRADIENT':GradientBoostingRegressor()
        }
        self.params ={
            "SVR":{
                'kernel':['linear','poly','rbf'],
                'gamma':['auto','scale'],
                'C':[1,5,10]
            },
            "GRADIENT":{
                'loss': ['squared_error', 'absolute_error'],
                'learning_rate':[0.02,0.05,0.1]
            }
        }
    def grid_trinning (self,X,Y):
        best_score=float("inf")
        best_model=None
        for name,reggresor in self.reggresors.items():
            grid_search=GridSearchCV(reggresor,self.params[name],cv=4).fit(X,Y.values.ravel())
            score=np.abs(grid_search.best_score_)            
            print(f"name:{name},score:{score}")
            if score< best_score: #the metric is going to be use her is  error but this dont  alaing with bigger better
                best_score=score
                print(f"new best _score{best_score}, new best model{name}")
                best_model=grid_search.best_estimator_
                print(type(grid_search.best_estimator_))
        utils=Utils()
        print("exporting new best model")
        utils.model_export(best_model,best_score)