import numpy as np
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
import os

from sklearn.feature_extraction.text import TfidfVectorizer

import pickle

import warnings
warnings.filterwarnings("ignore")


class User_Recommendation:
    def __init__(self):
       
        self.clean_df = pd.read_pickle("./savedData/preprocessed-dataframe.pkl")
        self.clean_df_recommended = self.clean_df[['id','name','reviews_complete_text', 'user_sentiment']]

        self.user_final_rating = pd.read_pickle("savedData/user_final_rating.pkl")

        file = open("./savedData/tfidf-vectorizer.pkl",'rb')
        self.vectorizer = pickle.load(file)
        file.close()

        # XGBoost
        file = open("./savedData/models/xgboost_classifier_20241114-105221.pkl",'rb')
        self.model = pickle.load(file)
        file.close()

    def get_top5_user_recommendations(self, user):
        if user in self.user_final_rating.index:
            # get the top 20  recommedation using the user_final_rating
            top20_reco = list(self.user_final_rating.loc[user].sort_values(ascending=False)[0:20].index)
            # get the product recommedation using the orig data used for trained model
            common_top20_reco = self.clean_df_recommended[self.clean_df['id'].isin(top20_reco)]
            # Apply the TFIDF Vectorizer for the given 20 products to convert data in reqd format for modeling
            X =  self.vectorizer.transform(common_top20_reco['reviews_complete_text'].values.astype(str))

            # Using the model from param to predict
            self.model.set_test_data(X)
            common_top20_reco['sentiment_pred']= self.model.predict()

            # Create a new dataframe "pred_df" to store the count of positive user sentiments
            temp_df = common_top20_reco.groupby(by='name').sum()
            # Create a new dataframe "pred_df" to store the count of positive user sentiments
            sent_df = temp_df[['sentiment_pred']]
            sent_df.columns = ['positive_sentiment_count']
            # Create a column to measure the total sentiment count
            sent_df['total_sentiment_count'] = common_top20_reco.groupby(by='name')['sentiment_pred'].count()
            # Calculate the positive sentiment percentage
            sent_df['positive_sentiment_percent'] = np.round(sent_df['positive_sentiment_count']/sent_df['total_sentiment_count']*100,2)
            # Return top 5 recommended products to the user
            result = sent_df.sort_values(by='positive_sentiment_percent', ascending=False)[:5]
            return result
        else:
            print(f"User name {user} doesn't exist")


def get_top5_recommendations(user):
    user_recommendation = User_Recommendation()
    recommendation_df = user_recommendation.get_top5_user_recommendations(user=user).reset_index()
    result = recommendation_df.to_dict("records")
    print(result)
    return result


def get_all_users():
    clean_df = pd.read_pickle("./savedData/preprocessed-dataframe.pkl")
    return clean_df[["reviews_username"]].to_dict("records")
    

def get_best_recommendation_users():
    best_recommendation_users = pd.read_pickle("./savedData/best_recommendation_users.pkl")
    return best_recommendation_users[["user"]].to_dict("records")