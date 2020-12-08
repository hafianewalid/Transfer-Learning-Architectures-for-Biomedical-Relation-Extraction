import copy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_fscore_support
from Train import *
from Load_data import *


class Model(BaseEstimator,TransformerMixin):
    def __init__(self,train_param,train_func):
        self.model=None
        self.train_func=train_func
        self.train_param=train_param

    def fit(self,X,Y):
        param=copy.deepcopy(self.train_param)
        param['X_train'],param['Y_train']=X,Y
        self.model,_=self.train_func(**param)

    def predict(self,X):
        pred = prediction(self.model,X)
        return pred


def cross_validation(train_param,train_func,K):
    estimator =Model(train_param,train_func)
    X,Y=train_param['X_train'],train_param['Y_train']
    pred = cross_val_predict(estimator=estimator,X=X,y=Y,cv=K,n_jobs=1)
    return pred,Y

def param_selection(list_train_param,train_func,K):
        F_type = global_param.traning_param['F_type']
        result=[]
        for train_param in list_train_param:
            pred,Y=cross_validation(train_param,train_func,K)
            rap=precision_recall_fscore_support(y_pred=pred,y_true=Y, average=F_type)
            result.append(rap)
        list_f=result[:][2]
        best_param=list_train_param[list_f.index(max(list_f))]
        return result,best_param
