from flask import Flask, jsonify, request
import SentimentAnalysisBusinessLogic

app = Flask(__name__)

@app.route('/get_recommendation')
def get_recommendation():
    user = request.args.get("user")
    result = SentimentAnalysisBusinessLogic.get_top5_recommendations(user=user)
    json_result = jsonify(result)
    return json_result

@app.route('/get_users')
def get_users():
    result = SentimentAnalysisBusinessLogic.get_all_users()
    response_result = {"count":len(result), "users": result}
    json_result = jsonify(response_result)
    return json_result

@app.route('/get_best_users')
def get_best_users():
    result = SentimentAnalysisBusinessLogic.get_best_recommendation_users()
    response_result = {"count":len(result), "users": result}
    json_result = jsonify(response_result)
    return json_result

if __name__ == '__main__':
    app.run(debug=True)