import pickle
import pandas as pd
import datetime


# Calculates the percentage of missing values in each column of a DataFrame.
# Args:
#   X (pandas.DataFrame): The input DataFrame.
# Returns:
#   pandas.DataFrame: A DataFrame containing the percentage of missing values for each column, sorted in descending order.
def get_missing_value_percentage(X):
  percent_missing = round((X.isnull().sum() / X.isnull().count()*100),3).to_frame('missing_percentage').sort_values('missing_percentage',ascending = False)
  return percent_missing


# Saves a DataFrame to a pickle file.
# Args:
#   df (pandas.DataFrame): The DataFrame to be saved.
def save_dataframe(df):
   df.to_pickle()


# Saves an object to a pickle file.
# Args:
#   obj: The object to be saved.
#   filename (str): The desired filename for the pickle file.
def save_object(obj, filename):
    pkl_filename = "savedData/" + filename + ".pkl"
    with open(pkl_filename, 'wb') as files:
      pickle.dump(obj, files)

# Saves a machine learning model object to a pickle file with a timestamp.
# Args:
#   obj: The machine learning model object to be saved.
#   filename (str): The desired base filename for the pickle file.
# This function saves the model object to a pickle file with the following format:
# - Path: "savedData/models/"
# - Filename:
#     - Base filename provided by the user (e.g., "my_model")
#     - Underscore separator ("_")
#     - Timestamp in YYYYMMDD-HHMMSS format (e.g., "20241118-143923")
#     - ".pkl" extension
# Example:
#     save_model(my_model, "logistic_regression")  # Saves to "savedData/models/logistic_regression_20241118-143923.pkl"
def save_model(obj, filename):
    pkl_filename = "savedData/models/" + filename + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".pkl"
    with open(pkl_filename, 'wb') as files:
      pickle.dump(obj, files)


# Gets a list of column values whose cumulative frequency is less than or equal to a given threshold.
# Args:
#   df (pandas.DataFrame): The input DataFrame.
#   column_name (str): The name of the column to analyze.
#   threshold (float): The threshold value for cumulative frequency.
# Returns:
#   list: A list of column values whose cumulative frequency is less than or equal to the threshold.
def get_columns_with_cumfreq(df, column_name,threshold):
    df_cat_freq = df[column_name].value_counts()
    df_cat_freq = pd.DataFrame({'column':df_cat_freq.index, 'value':df_cat_freq.values})
    df_cat_freq['perc'] = df_cat_freq['value'].cumsum()/df_cat_freq['value'].sum()
    return list(df_cat_freq.loc[df_cat_freq['perc']<=threshold].column)