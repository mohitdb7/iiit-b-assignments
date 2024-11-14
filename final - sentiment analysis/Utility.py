import pickle
import pandas as pd
import datetime


def get_missing_value_percentage(X):
  percent_missing = round((X.isnull().sum() / X.isnull().count()*100),3).to_frame('missing_percentage').sort_values('missing_percentage',ascending = False)
  return percent_missing

def save_dataframe(df):
   df.to_pickle()

#To save objects at intermediate step
def save_object(obj, filename):
    pkl_filename = "savedData/" + filename + ".pkl"
    with open(pkl_filename, 'wb') as files:
      pickle.dump(obj, files)

def save_model(obj, filename):
    pkl_filename = "savedData/models/" + filename + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".pkl"
    with open(pkl_filename, 'wb') as files:
      pickle.dump(obj, files)

def get_columns_with_cumfreq(df, column_name,threshold):
    df_cat_freq = df[column_name].value_counts()
    df_cat_freq = pd.DataFrame({'column':df_cat_freq.index, 'value':df_cat_freq.values})
    df_cat_freq['perc'] = df_cat_freq['value'].cumsum()/df_cat_freq['value'].sum()
    return list(df_cat_freq.loc[df_cat_freq['perc']<=threshold].column)