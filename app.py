from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
import joblib

pipe = joblib.load('model.pkl')

teams = ['Sunrisers Hyderabad','Mumbai Indians','Royal Challengers Bangalore',
         'Kolkata Knight Riders','Kings XI Punjab','Chennai Super Kings',
         'Rajasthan Royals','Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        batting_team = request.form['batting_team']
        bowling_team = request.form['bowling_team']
        city = request.form['city']
        target = int(request.form['target'])
        score = int(request.form['score'])
        overs = float(request.form['overs'])
        wickets = int(request.form['wickets'])

        runs_left = target - score
        balls_left = 120 - (overs * 6)
        wickets_left = 10 - wickets
        crr = score / overs if overs != 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left != 0 else 0

        input_df = pd.DataFrame({
            'batting_team':[batting_team],
            'bowling_team':[bowling_team],
            'city':[city],
            'runs_left':[runs_left],
            'balls_left':[balls_left],
            'wickets':[wickets_left],
            'total_runs_x':[target],
            'crr':[crr],
            'rrr':[rrr]
        })

        result = pipe.predict_proba(input_df)
        loss = round(result[0][0] * 100)
        win = round(result[0][1] * 100)

        return render_template('index.html', 
                               teams=teams, 
                               cities=cities,
                               batting_team=batting_team,
                               bowling_team=bowling_team,
                               win=win,
                               loss=loss)

    return render_template('index.html', teams=teams, cities=cities)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)