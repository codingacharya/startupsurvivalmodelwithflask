from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load('startup_model.pkl')

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = {
        'team_experience_years': int(request.form['team_experience_years']),
        'past_successes': int(request.form['past_successes']),
        'funding_rounds': int(request.form['funding_rounds']),
        'total_funding': float(request.form['total_funding']),
        'has_top_investor': request.form['has_top_investor'],
        'debt_ratio': float(request.form['debt_ratio']),
        'cash_flow_score': float(request.form['cash_flow_score']),
        'industry': request.form['industry'],
        'regulatory_risk_score': float(request.form['regulatory_risk_score']),
        'customer_retention_rate': float(request.form['customer_retention_rate']),
        'customer_satisfaction_score': float(request.form['customer_satisfaction_score'])
    }

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    result = "✅ The startup is likely to SURVIVE!" if prediction == 1 else "❌ The startup is at risk of failure."
    return f"<h2>{result}</h2><p>Survival Probability: {prob:.2%}</p><a href='/'>Try another</a>"

if __name__ == '__main__':
    app.run(debug=True)
