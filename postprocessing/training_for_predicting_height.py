import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse


def get_height_reg(name='car', folds=5):
    
    wlh_info = pd.read_csv("/media/jionie/my_disk/Kaggle/Lyft/codes/postprocessing/wlh_info.csv")
    kf = KFold(n_splits=folds)
    
    wlh = wlh_info[wlh_info['name'] == name]
    X = np.concatenate((np.expand_dims(wlh.width.values, axis=1), \
                        np.expand_dims(wlh.length.values, axis=1)), axis=1)
    y = wlh.height.values
    class_height = {'animal':0.51,'bicycle':1.44,'bus':3.44,'car':1.72,'emergency_vehicle':2.39,'motorcycle':1.59,
                    'other_vehicle':3.23,'pedestrian':1.78,'truck':3.44}
    loss = []
    loss_avg = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        reg = linear_model.BayesianRidge()
    #     kernel = DotProduct() + WhiteKernel()
    #     reg = GaussianProcessRegressor(kernel=kernel, random_state=42)
        reg.fit(X_train, y_train)
        
        y_pred = reg.predict(X_test)
        
        loss.append(mse(y_test, y_pred))
        loss_avg.append(mse(y_test, np.ones_like(y_test) * class_height[name]))
        
    print("For class: ", name, folds, "validation", "using predicted height mse: ", np.mean(loss), "using average height mse: ", np.mean(loss_avg))
    
    reg.fit(X, y)
    
    return reg

def get_height_reg_by_length(name='car', folds=5):
    
    wlh_info = pd.read_csv("/media/jionie/my_disk/Kaggle/Lyft/codes/postprocessing/wlh_info.csv")
    kf = KFold(n_splits=folds)
    
    wlh = wlh_info[wlh_info['name'] == name]
    X = np.expand_dims(wlh.length.values, axis=1)
    y = wlh.height.values
    class_height = {'animal':0.51,'bicycle':1.44,'bus':3.44,'car':1.72,'emergency_vehicle':2.39,'motorcycle':1.59,
                    'other_vehicle':3.23,'pedestrian':1.78,'truck':3.44}
    loss = []
    loss_avg = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        reg = linear_model.BayesianRidge()
    #     kernel = DotProduct() + WhiteKernel()
    #     reg = GaussianProcessRegressor(kernel=kernel, random_state=42)
        reg.fit(X_train, y_train)
        
        y_pred = reg.predict(X_test)
        
        loss.append(mse(y_test, y_pred))
        loss_avg.append(mse(y_test, np.ones_like(y_test) * class_height[name]))
        
    print("For class: ", name, folds, "validation", "using predicted height mse: ", np.mean(loss), "using average height mse: ", np.mean(loss_avg))
    
    reg.fit(X, y)
    
    return reg
