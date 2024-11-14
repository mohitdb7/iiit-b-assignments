import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier


class ModelFactory:
  def __init__(self,model, model_name, X_train,y_train,X_test,y_test):
    self.model = model
    self.model_name = model_name
    self.X_train = X_train
    self.y_train = y_train
    self.X_test = X_test
    self.y_test = y_test
    self.train_metrics = []
    self.test_metrics = []

  def train(self):
    self.model.fit(self.X_train,self.y_train)
    return self.model.predict(self.X_train)

  def test(self):
    return self.model.predict(self.X_test)

  # This API is added for model load cases
  def set_test_data(self,test_data):
    self.X_test = test_data

  def predict(self):
    return self.model.predict(self.X_test)


  def evaluate_metrics_train(self,y_pred):
    accuracy = round(accuracy_score(self.y_train, y_pred),2)
    precision = round(precision_score(self.y_train,y_pred),2)
    recall = round(recall_score(self.y_train,y_pred),2)
    f1 = round(f1_score(self.y_train,y_pred),2)
    auc_score = round(roc_auc_score(self.y_train,y_pred),2)
    self.train_metrics.append(accuracy)
    self.train_metrics.append(precision)
    self.train_metrics.append(recall)
    self.train_metrics.append(f1)
    self.train_metrics.append(auc_score)

    print("Train Data Metrics - ", self.model_name)
    
    print("*"*20)
    print("*"*20)
    print("Accuracy:", self.train_metrics[0])
    print("Precision:", self.train_metrics[1])
    print("Recall:", self.train_metrics[2])
    print("F1 Score:", self.train_metrics[3])
    print("AUC Score:", self.train_metrics[4])
    
    print("*"*20)
    print("*"*20)
    self.get_confusion_matrix_train(y_pred)

    print("*"*20)
    print("*"*20)
    self.plot_roc_curve(self.y_train, y_pred)
    return


  def evaluate_metrics_test(self,y_pred):
    accuracy = round(accuracy_score(self.y_test, y_pred),2)
    precision = round(precision_score(self.y_test,y_pred),2)
    recall = round(recall_score(self.y_test,y_pred),2)
    f1 = round(f1_score(self.y_test,y_pred),2)
    auc_score = round(roc_auc_score(self.y_test,y_pred),2)

    self.test_metrics.append(accuracy)
    self.test_metrics.append(precision)
    self.test_metrics.append(recall)
    self.test_metrics.append(f1)
    self.test_metrics.append(auc_score)
    

    print("Test Data Metrics - ", self.model_name)
    
    print("*"*20)
    print("*"*20)
    print("Accuracy:", self.test_metrics[0])
    print("Precision:", self.test_metrics[1])
    print("Recall:", self.test_metrics[2])
    print("F1 Score:", self.test_metrics[3])
    print("AUC Score:", self.test_metrics[4])
    
    print("*"*20)
    print("*"*20)
    self.get_confusion_matrix_test(y_pred)

    print("*"*20)
    print("*"*20)
    self.plot_roc_curve(self.y_test, y_pred)
    return


  def get_confusion_matrix_train(self, y_pred):
    confusion_mat = confusion_matrix(self.y_train, y_pred)
    self.plot_confusion_matrix(confusion_mat,[0,1])
    return


  def get_confusion_matrix_test(self, y_pred):
    confusion_mat = confusion_matrix(self.y_test, y_pred)
    self.plot_confusion_matrix(confusion_mat,[0,1])
    return

  def plot_confusion_matrix(self, data, labels):
      sns.set(color_codes=True)
      plt.title("Confusion Matrix")
      ax = sns.heatmap(data, annot=True, cmap="Blues", fmt=".1f")
      ax.set_xticklabels(labels)
      ax.set_yticklabels(labels)
      ax.set(ylabel="True Values", xlabel="Predicted Values")
      plt.show()
      return

  def get_train_metrics(self):
    return self.train_metrics


  def get_test_metrics(self):
    return self.test_metrics
  
  def plot_roc_curve(self, y_actual, y_pred):
        fpr, tpr, _ = metrics.roc_curve(y_actual,  y_pred)
        auc = metrics.roc_auc_score(y_actual, y_pred)
        plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
        plt.legend(loc=4)
        plt.show()
  


class LogisticRegressionModel:
    def __init__(self,model_name, X_train, y_train, X_test, y_test):
      self.X_train = X_train
      self.y_train = y_train
      self.X_test = X_test
      self.y_test = y_test
      self.model_name = model_name

      lr = LogisticRegression(random_state=42)
      self.lr_obj = ModelFactory(lr, self.model_name, self.X_train, self.y_train, self.X_test, self.y_test)
    
    def evaluate_train(self):
        # Train
        y_train_pred = self.lr_obj.train()
        self.lr_obj.evaluate_metrics_train(y_train_pred)

    def evaluate_test(self):
        # Train
        y_test_pred = self.lr_obj.test()
        self.lr_obj.evaluate_metrics_test(y_test_pred)


class NaiveBayesRegressionModel:
    def __init__(self,model_name, X_train, y_train, X_test, y_test):
      self.X_train = X_train
      self.y_train = y_train
      self.X_test = X_test
      self.y_test = y_test
      self.model_name = model_name

      nb = MultinomialNB()
      self.nb_obj = ModelFactory(nb, self.model_name, self.X_train, self.y_train, self.X_test, self.y_test)     

    def evaluate_train(self):
        # Train
        y_train_pred = self.nb_obj.train()
        self.nb_obj.evaluate_metrics_train(y_train_pred)

    def evaluate_test(self):
        # Train
        y_test_pred = self.nb_obj.test()
        self.nb_obj.evaluate_metrics_test(y_test_pred) 
       


class XGBoostRegressionModel:
    def __init__(self,model_name, X_train, y_train, X_test, y_test):
      self.X_train = X_train
      self.y_train = y_train
      self.X_test = X_test
      self.y_test = y_test
      self.model_name = model_name

      xgb_c = xgb.XGBClassifier(random_state=42, n_jobs=-1)
      self.xgb_c_obj = ModelFactory(xgb_c, self.model_name, self.X_train, self.y_train, self.X_test, self.y_test)

    def evaluate_train(self):
        # Train
        y_train_pred = self.xgb_c_obj.train()
        self.xgb_c_obj.evaluate_metrics_train(y_train_pred)

    def evaluate_test(self):
        # Train
        y_test_pred = self.xgb_c_obj.test()
        self.xgb_c_obj.evaluate_metrics_test(y_test_pred) 


class XGBoostRegressionHyperParameterModel:
    def __init__(self,model_name, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = model_name

        param_grid_xgb={'learning_rate': [0.0001, 0.001, 0.01,0.1, 1], 'max_depth': [5, 8, 10, 13, 15, 18, 20],
                        'n_estimators': [1, 3, 5, 7]}
        gv_xgb_hp_tuned = GridSearchCV(cv=5, estimator=xgb.XGBClassifier(random_state=42, n_jobs=-1),
                                param_grid=param_grid_xgb, verbose=1)
        xgb_hp_tuned = gv_xgb_hp_tuned.fit(X_train, y_train).best_estimator_
        self.xgb_hp_tuned_obj = ModelFactory(xgb_hp_tuned, self.model_name, self.X_train, self.y_train, self.X_test, self.y_test)

    def evaluate_train(self):
        # Train
        y_train_pred = self.xgb_hp_tuned_obj.train()
        self.xgb_hp_tuned_obj.evaluate_metrics_train(y_train_pred)

    def evaluate_test(self):
        # Train
        y_test_pred = self.xgb_hp_tuned_obj.test()
        self.xgb_hp_tuned_obj.evaluate_metrics_test(y_test_pred) 



class RandomForestClassifierModel:
    def __init__(self,model_name, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = model_name

        rf_classifer = RandomForestClassifier(random_state=42,n_jobs=-1)
        self.rf_obj = ModelFactory(rf_classifer, self.model_name, self.X_train, self.y_train, self.X_test, self.y_test)

    def evaluate_train(self):
        # Train
        y_train_pred = self.rf_obj.train()
        self.rf_obj.evaluate_metrics_train(y_train_pred)

    def evaluate_test(self):
        # Train
        y_test_pred = self.rf_obj.test()
        self.rf_obj.evaluate_metrics_test(y_test_pred)


class RandomForestClassifierHyperParameterModel:
    def __init__(self,model_name, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = model_name

        param_grid_rf = {'max_depth': [2, 3, 5, 10],
                 'min_samples_leaf': [5, 10, 20],
                  'n_estimators': [10, 25, 50, 100]}

        gv_rf_hp_tuned = GridSearchCV(cv=5, estimator=RandomForestClassifier(random_state=42,n_jobs=-1),
                            param_grid=param_grid_rf, verbose=1)
        rf_hp_tuned = gv_rf_hp_tuned.fit(X_train, y_train).best_estimator_
        self.rf_hp_tuned_obj = ModelFactory(rf_hp_tuned, self.model_name, self.X_train, self.y_train, self.X_test, self.y_test)

    def evaluate_train(self):
        # Train
        y_train_pred = self.rf_hp_tuned_obj.train()
        self.rf_hp_tuned_obj.evaluate_metrics_train(y_train_pred)

    def evaluate_test(self):
        # Train
        y_test_pred = self.rf_hp_tuned_obj.test()
        self.rf_hp_tuned_obj.evaluate_metrics_test(y_test_pred)     