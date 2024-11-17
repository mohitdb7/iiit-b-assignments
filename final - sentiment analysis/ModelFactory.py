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


#-----------------------------------------------------------------------------------------------------------------------#

#   This class provides a factory pattern for creating and evaluating machine learning models.
#   It simplifies the process of training, testing, and evaluating models by providing a consistent interface.
class ModelFactory:
  
  # Initializes the ModelFactory instance.
  # Args:
  #   model: The machine learning model object to be trained and evaluated.
  #   model_name: A string representing the name of the model.
  #   X_train: The training data features.
  #   y_train: The training data target labels.
  #   X_test: The testing data features.
  #   y_test: The testing data target labels.
  def __init__(self,model, model_name, X_train,y_train,X_test,y_test):
    self.model = model
    self.model_name = model_name
    self.X_train = X_train
    self.y_train = y_train
    self.X_test = X_test
    self.y_test = y_test
    self.train_metrics = []
    self.test_metrics = []

  # Trains the model on the provided training data.
  # Returns:
  #   The model's predictions on the training data.
  def train(self):
    self.model.fit(self.X_train,self.y_train)
    return self.model.predict(self.X_train)

  # Evaluates the model on the provided testing data.
  # Returns:
  #   The model's predictions on the testing data.
  def test(self):
    return self.model.predict(self.X_test)

  # Sets the testing data for the model.
  # This method allows you to update the testing data after the ModelFactory is initialized.
  # Args:
  #   test_data: The new testing data features.
  # This API is added for model load cases
  def set_test_data(self,test_data):
    self.X_test = test_data


  # Predicts on the testing data using the trained model.

  # Returns:
  #   The model's predictions on the testing data.
  def predict(self):
    return self.model.predict(self.X_test)


  # Calculates and stores various evaluation metrics for the training data.
  # Creates confusion metrics and roc_curve
  # Args:
  #   y_pred: The model's predictions on the training data.
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

  # Calculates and stores various evaluation metrics for the testing data.
  # Creates confusion metrics and roc_curve
  # Args:
  #   y_pred: The model's predictions on the testing data.
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

  # Generates and plots the confusion matrix for the training data.
  # This method calculates the confusion matrix using the `confusion_matrix` function
  # from scikit-learn and then calls the `plot_confusion_matrix` method to visualize it.
  # Args:
  #   y_pred: The model's predictions on the training data.
  def get_confusion_matrix_train(self, y_pred):
    confusion_mat = confusion_matrix(self.y_train, y_pred)
    self.plot_confusion_matrix(confusion_mat,[0,1])
    return


  # Generates and plots the confusion matrix for the testing data.
  # This method is similar to `get_confusion_matrix_train` but uses the testing data
  # (self.y_test) to calculate the confusion matrix.
  # Args:
  #   y_pred: The model's predictions on the testing data.
  def get_confusion_matrix_test(self, y_pred):
    confusion_mat = confusion_matrix(self.y_test, y_pred)
    self.plot_confusion_matrix(confusion_mat,[0,1])
    return

  # Visualizes the confusion matrix using Seaborn's heatmap.
  # This method takes the confusion matrix data and labels as input and creates a heatmap
  # using Seaborn. It sets the title, labels, and formatting for the plot.
  # Args:
  #   data: The confusion matrix data.
  #   labels: The labels for the confusion matrix (e.g., class labels).
  def plot_confusion_matrix(self, data, labels):
      sns.set(color_codes=True)
      plt.title("Confusion Matrix")
      ax = sns.heatmap(data, annot=True, cmap="Blues", fmt=".1f")
      ax.set_xticklabels(labels)
      ax.set_yticklabels(labels)
      ax.set(ylabel="True Values", xlabel="Predicted Values")
      plt.show()
      return

  # Returns the list of evaluation metrics for the training data.
  # This method simply returns the `self.train_metrics` list, which stores the calculated
  # metrics (accuracy, precision, etc.) for the training data.
  def get_train_metrics(self):
    return self.train_metrics


  # Returns the list of evaluation metrics for the testing data.
  # This method is similar to `get_train_metrics` but returns the `self.test_metrics` list
  # containing the metrics for the testing data.
  def get_test_metrics(self):
    return self.test_metrics


  # Plots the Receiver Operating Characteristic (ROC) curve for the model.
  # This method calculates the False Positive Rate (FPR), True Positive Rate (TPR), and Area Under 
  # the Curve (AUC) using scikit-learn's `roc_curve` and `roc_auc_score` functions. It then creates
  # a plot of the ROC curve using Matplotlib.
  # Args:
  #   y_actual: The ground truth labels.
  #   y_pred: The model's predictions.
  def plot_roc_curve(self, y_actual, y_pred):
        fpr, tpr, _ = metrics.roc_curve(y_actual,  y_pred)
        auc = metrics.roc_auc_score(y_actual, y_pred)
        plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
        plt.legend(loc=4)
        plt.show()
  
#-----------------------------------------------------------------------------------------------------------------------#


# This class encapsulates a Logistic Regression model, leveraging the ModelFactory class for training, testing, and evaluation.
# Attributes:
#     X_train (array-like): The training data features.
#     y_train (array-like): The training data target labels.
#     X_test (array-like): The testing data features.
#     y_test (array-like): The testing data target labels.
#     model_name (str): The name of the model.
#     lr_obj (ModelFactory): An instance of the ModelFactory class for model operations.
class LogisticRegressionModel:
    
    # Initializes the LogisticRegressionModel instance.
    # Args:
    #     model_name (str): The name of the model.
    #     X_train (array-like): The training data features.
    #     y_train (array-like): The training data target labels.
    #     X_test (array-like): The testing data features.
    #     y_test (array-like): The testing data target labels.
    def __init__(self,model_name, X_train, y_train, X_test, y_test):
      self.X_train = X_train
      self.y_train = y_train
      self.X_test = X_test
      self.y_test = y_test
      self.model_name = model_name

      lr = LogisticRegression(random_state=42)
      self.lr_obj = ModelFactory(lr, self.model_name, self.X_train, self.y_train, self.X_test, self.y_test)
    

    # Trains the Logistic Regression model and evaluates its performance on the training data.
    # This method trains the model using the ModelFactory's `train` method, generates predictions on the training data,
    # and then evaluates the model's performance using the `evaluate_metrics_train` method of the ModelFactory.
    def evaluate_train(self):
        y_train_pred = self.lr_obj.train()
        self.lr_obj.evaluate_metrics_train(y_train_pred)


    # Evaluates the Logistic Regression model on the testing data.
    # This method generates predictions on the testing data using the ModelFactory's `test` method and
    # then evaluates the model's performance using the `evaluate_metrics_test` method of the ModelFactory.
    def evaluate_test(self):
        y_test_pred = self.lr_obj.test()
        self.lr_obj.evaluate_metrics_test(y_test_pred)


#-----------------------------------------------------------------------------------------------------------------------#



# This class encapsulates a Naive Bayes Regression model, leveraging the ModelFactory class for training, testing, and evaluation.
# Attributes:
#     X_train (array-like): The training data features.
#     y_train (array-like): The training data target labels.
#     X_test (array-like): The testing data features.
#     y_test (array-like): The testing data target labels.
#     model_name (str): The name of the model.
#     nb_obj (ModelFactory): An instance of the ModelFactory class for model operations.
class NaiveBayesRegressionModel:
    
    # Initializes the NaiveBayesRegressionModel instance.
    # Args:
    #     model_name (str): The name of the model.
    #     X_train (array-like): The training data features.
    #     y_train (array-like): The training data target labels.
    #     X_test (array-like): The testing data features.
    #     y_test (array-like): The testing data target labels.
    def __init__(self,model_name, X_train, y_train, X_test, y_test):
      self.X_train = X_train
      self.y_train = y_train
      self.X_test = X_test
      self.y_test = y_test
      self.model_name = model_name

      nb = MultinomialNB()
      self.nb_obj = ModelFactory(nb, self.model_name, self.X_train, self.y_train, self.X_test, self.y_test)     


    # Trains the Naive Bayes Regression model and evaluates its performance on the training data.
    # This method trains the model using the ModelFactory's `train` method, generates predictions on the training data,
    # and then evaluates the model's performance using the `evaluate_metrics_train` method of the ModelFactory.
    def evaluate_train(self):
        y_train_pred = self.nb_obj.train()
        self.nb_obj.evaluate_metrics_train(y_train_pred)

    # Evaluates the Naive Bayes Regression model on the testing data.
    # This method generates predictions on the testing data using the ModelFactory's `test` method and
    # then evaluates the model's performance using the `evaluate_metrics_test` method of the ModelFactory.
    def evaluate_test(self):
        y_test_pred = self.nb_obj.test()
        self.nb_obj.evaluate_metrics_test(y_test_pred) 
       

#-----------------------------------------------------------------------------------------------------------------------#



# This class encapsulates a XGBoost Regression Model, leveraging the ModelFactory class for training, testing, and evaluation.
# Attributes:
#     X_train (array-like): The training data features.
#     y_train (array-like): The training data target labels.
#     X_test (array-like): The testing data features.
#     y_test (array-like): The testing data target labels.
#     model_name (str): The name of the model.
#     xgb_c_obj (ModelFactory): An instance of the ModelFactory class for model operations.
class XGBoostRegressionModel:
    
    # Initializes the XGBoostRegressionModel instance.
    # Args:
    #     model_name (str): The name of the model.
    #     X_train (array-like): The training data features.
    #     y_train (array-like): The training data target labels.
    #     X_test (array-like): The testing data features.
    #     y_test (array-like): The testing data target labels.
    def __init__(self,model_name, X_train, y_train, X_test, y_test):
      self.X_train = X_train
      self.y_train = y_train
      self.X_test = X_test
      self.y_test = y_test
      self.model_name = model_name

      xgb_c = xgb.XGBClassifier(random_state=42, n_jobs=-1)
      self.xgb_c_obj = ModelFactory(xgb_c, self.model_name, self.X_train, self.y_train, self.X_test, self.y_test)


    # Trains the XGBoost Regression model and evaluates its performance on the training data.
    # This method trains the model using the ModelFactory's `train` method, generates predictions on the training data,
    # and then evaluates the model's performance using the `evaluate_metrics_train` method of the ModelFactory.
    def evaluate_train(self):
        y_train_pred = self.xgb_c_obj.train()
        self.xgb_c_obj.evaluate_metrics_train(y_train_pred)


    # Evaluates the XGBoost Regression model on the testing data.
    # This method generates predictions on the testing data using the ModelFactory's `test` method and
    # then evaluates the model's performance using the `evaluate_metrics_test` method of the ModelFactory.
    def evaluate_test(self):
        y_test_pred = self.xgb_c_obj.test()
        self.xgb_c_obj.evaluate_metrics_test(y_test_pred) 



#-----------------------------------------------------------------------------------------------------------------------#




# This class performs hyperparameter tuning for an XGBoost regression model and leverages the ModelFactory class for training, testing, and evaluation.
#   Attributes:
#       X_train (array-like): The training data features.
#       y_train (array-like): The training data target labels.
#       X_test (array-like): The testing data features.
#       y_test (array-like): The testing data target labels.
#       model_name (str): The name of the model.
#       xgb_hp_tuned_obj (ModelFactory): An instance of the ModelFactory class for the hyper-tuned XGBoost model.
class XGBoostRegressionHyperParameterModel:
    
    # Initializes the XGBoostRegressionHyperParameterModel instance.
    # Args:
    #     model_name (str): The name of the model.
    #     X_train (array-like): The training data features.
    #     y_train (array-like): The training data target labels.
    #     X_test (array-like): The testing data features.
    #     y_test (array-like): The testing data target labels.
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


    # Trains the hyper-tuned XGBoost model and evaluates its performance on the training data.
    # This method trains the tuned model using the ModelFactory's `train` method, generates predictions on the training data,
    # and then evaluates the model's performance using the `evaluate_metrics_train` method of the ModelFactory.
    def evaluate_train(self):
        y_train_pred = self.xgb_hp_tuned_obj.train()
        self.xgb_hp_tuned_obj.evaluate_metrics_train(y_train_pred)


    # Evaluates the hyper-tuned XGBoost model on the testing data.
    # This method generates predictions on the testing data using the ModelFactory's `test` method and
    # then evaluates the model's performance using the `evaluate_metrics_test` method of the ModelFactory.
    def evaluate_test(self):
        y_test_pred = self.xgb_hp_tuned_obj.test()
        self.xgb_hp_tuned_obj.evaluate_metrics_test(y_test_pred) 


#-----------------------------------------------------------------------------------------------------------------------#


# This class encapsulates a Random Forest Classifier Model, leveraging the ModelFactory class for training, testing, and evaluation.
# Attributes:
#     X_train (array-like): The training data features.
#     y_train (array-like): The training data target labels.
#     X_test (array-like): The testing data features.
#     y_test (array-like): The testing data target labels.
#     model_name (str): The name of the model.
#     rf_obj (ModelFactory): An instance of the ModelFactory class for model operations.
class RandomForestClassifierModel:
    
    # Initializes the RandomForestClassifierModel instance.
    # Args:
    #     model_name (str): The name of the model.
    #     X_train (array-like): The training data features.
    #     y_train (array-like): The training data target labels.
    #     X_test (array-like): The testing data features.
    #     y_test (array-like): The testing data target labels.
    def __init__(self,model_name, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = model_name

        rf_classifer = RandomForestClassifier(random_state=42,n_jobs=-1)
        self.rf_obj = ModelFactory(rf_classifer, self.model_name, self.X_train, self.y_train, self.X_test, self.y_test)


    # Trains the XGBoost Regression model and evaluates its performance on the training data.
    # This method trains the model using the ModelFactory's `train` method, generates predictions on the training data,
    # and then evaluates the model's performance using the `evaluate_metrics_train` method of the ModelFactory.
    def evaluate_train(self):
        y_train_pred = self.rf_obj.train()
        self.rf_obj.evaluate_metrics_train(y_train_pred)


    # Evaluates the Naive Bayes Regression model on the testing data.
    # This method generates predictions on the testing data using the ModelFactory's `test` method and
    # then evaluates the model's performance using the `evaluate_metrics_test` method of the ModelFactory.
    def evaluate_test(self):
        y_test_pred = self.rf_obj.test()
        self.rf_obj.evaluate_metrics_test(y_test_pred)


#-----------------------------------------------------------------------------------------------------------------------#


# This class performs hyperparameter tuning for a Random Forest Classifier model and leverages the ModelFactory class for training, testing, and evaluation.
# Attributes:
#     X_train (array-like): The training data features.
#     y_train (array-like): The training data target labels.
#     X_test (array-like): The testing data features.
#     y_test (array-like): The testing data target labels.
#     model_name (str): The name of the model.
#     rf_hp_tuned_obj (ModelFactory): An instance of the ModelFactory class for the hyper-tuned Random Forest Classifier model.
class RandomForestClassifierHyperParameterModel:
    
    # Initializes the RandomForestClassifierHyperParameterModel instance.
    # Args:
    #     model_name (str): The name of the model.
    #     X_train (array-like): The training data features.
    #     y_train (array-like): The training data target labels.
    #     X_test (array-like): The testing data features.
    #     y_test (array-like): The testing data target labels.
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


    # Trains the hyper-tuned Random Forest Classifier model and evaluates its performance on the training data.
    # This method trains the tuned model using the ModelFactory's `train` method, generates predictions on the training data,
    # and then evaluates the model's performance using the `evaluate_metrics_train` method of the ModelFactory.
    def evaluate_train(self):
        y_train_pred = self.rf_hp_tuned_obj.train()
        self.rf_hp_tuned_obj.evaluate_metrics_train(y_train_pred)


    # Evaluates the hyper-tuned Random Forest Classifier model on the testing data.
    # This method generates predictions on the testing data using the ModelFactory's `test` method and
    # then evaluates the model's performance using the `evaluate_metrics_test` method of the ModelFactory.
    def evaluate_test(self):
        y_test_pred = self.rf_hp_tuned_obj.test()
        self.rf_hp_tuned_obj.evaluate_metrics_test(y_test_pred)     