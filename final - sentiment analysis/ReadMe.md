# Task 8: Deployment

## **How to Run the Server**
Run command -> `python app_root.py` to start the server
Two files are there
- app_root.py
- SentimentAnalysisBusinessLogic.py

It has three APIs
- /get_recommendation?user=<username>
This service will return the top 5 recommendations
- /get_users
This service will return the list of users
- /get_best_users
This service will return 10 users that has best recommendations


## Where can you find the complete sentiment analysis
- `Mohit-SentimentAnalysis.ipynb` contains the complete assignment

## Where are all the models and intermediate files
- They are present inside the savedData folder


## **Important Note:** Only XGBoost model pickel is kept as upload was having a 50MB upload limit
