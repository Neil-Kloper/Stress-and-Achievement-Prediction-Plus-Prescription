from flask import Flask,request, url_for, redirect, render_template, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)


model = keras.models.load_model('./model/wellbeing_model')
x_columns = [
    'FRUITS_VEGGIES',
    'PLACES_VISITED',
    'CORE_CIRCLE',
    'SUPPORTING_OTHERS',
    'SOCIAL_NETWORK',
    'DONATION',
    'TODO_COMPLETED',
    'FLOW',
    'DAILY_STEPS',
    'LIVE_VISION',
    'SLEEP_HOURS',
    'LOST_VACATION',
    'DAILY_SHOUTING',
    'PERSONAL_AWARDS',
    'TIME_FOR_PASSION',
    'WEEKLY_MEDITATION',
    'AGE'
]

wellbeing_stats = {
    'FRUITS_VEGGIES': (2.9226723436228164, 1.4427392618232717),
    'DAILY_STRESS': (2.7916849289336922, 1.3678007467520172),
    'PLACES_VISITED': (5.233235238870453, 3.3118466202433825),
    'CORE_CIRCLE': (5.508296287020224, 2.8402868211101433),
    'SUPPORTING_OTHERS': (5.616179325026611, 3.2419369588232674),
    'SOCIAL_NETWORK': (6.474046709661261, 3.08664272775673),
    'ACHIEVEMENT': (4.000688748356396, 2.7559123263254053),
    'DONATION': (2.715171247886795, 1.851556132858851),
    'BMI_RANGE': (1.410619247385887, 0.491961619609354),
    'TODO_COMPLETED': (5.745977083463778, 2.6241786624245114),
    'FLOW': (3.194477490451443, 2.3572846752920262),
    'DAILY_STEPS': (5.703587752801954, 2.8911020666495197),
    'LIVE_VISION': (3.7521758186713416, 3.2310825267930916),
    'SLEEP_HOURS': (7.042952852044331, 1.199053434423808),
    'LOST_VACATION': (2.8984409241750675, 3.6918674791997406),
    'DAILY_SHOUTING': (2.930999937386513, 2.6763413227911825),
    'SUFFICIENT_INCOME': (1.7289462150147141, 0.4445177193376449),
    'PERSONAL_AWARDS': (5.711289211696199, 3.0895399360748805),
    'TIME_FOR_PASSION': (3.3262788804708534, 2.729127663833442),
    'WEEKLY_MEDITATION': (6.233610919792123, 3.0164794474060197),
    'AGE': (2.602091290463966, 0.9444260282900208),
    'GENDER': (0.6172437543046773, 0.48607478404407434),
    'WORK_LIFE_BALANCE_SCORE': (666.75051029992, 45.0211030065477)
}

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    list_pred = [int(x) for x in request.form.values()]
    pred_dict = {}
    pred_dict['Actual'] = list_pred
    idx = 0
    for num in list_pred:
        if num < 10:
            plus_1 = list_pred.copy()
            plus_1[idx] = num + 1
            label = str(x_columns[idx]) + ' plus 1'
            pred_dict[label] = plus_1
        if num > 0:
            minus_1 = list_pred.copy()
            minus_1[idx] = num - 1
            label = str(x_columns[idx]) + ' minus 1'
            pred_dict[label] = minus_1
        idx+=1
    personal_preds_batch = pd.DataFrame.from_dict(pred_dict, orient='index', columns = x_columns)

    for col in x_columns:
        personal_preds_batch[col] = (personal_preds_batch[col]-wellbeing_stats[col][0])/wellbeing_stats[col][1]

    
    personal_preds_batch[['DAILY_STRESS', 'ACHIEVEMENT']] = model.predict(personal_preds_batch)
    personal_preds_batch = personal_preds_batch[['DAILY_STRESS', 'ACHIEVEMENT']]
    result = personal_preds_batch.to_html()
    return render_template('home.html',pred=result)

# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     data = request.get_json(force=True)
#     data_unseen = pd.DataFrame([data])
#     prediction = predict_model(model, data=data_unseen)
#     output = prediction.Label[0]
#     return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
