from flask import Flask, request, render_template, redirect, url_for, session

from functions import parse_weather_question, get_weather_data, generate_weather_response,create_temperature_visualisation
import logging
import os 
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import plotly.express as px

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'a_default_secret_key_for_testing') 
app.config['DEBUG'] = True
app.logger.setLevel(logging.INFO)

@app.route('/', methods=['GET'])
@app.route('/home', methods=['GET'])
def home():
    response = session.pop('weather_response', None)
    full_path = session.pop('full_path', None)
    new_path = session.pop('new_path', None)
    return render_template('index.html', response=response,full_path=full_path,new_path=new_path)


@app.route('/get_response', methods=['POST', 'GET'])
def get_response():
    if request.method == 'GET':
        return redirect(url_for('home'))
    df = pd.DataFrame()
    query = request.form.get('weather')
    if not query:
        response = "Query field cannot be empty. Please ask a question about the weather."
    else:
        main_query = parse_weather_question(query)
        for i in main_query['Location']:
            location = i
            df_1,weather_data = get_weather_data(location)
            df = pd.concat([df, df_1])
        main_query['Location'].pop(0)
        response = generate_weather_response(main_query, weather_data)
        full_path,new_path = create_temperature_visualisation(df)

    session['weather_response'] = response
    session['full_path'] = full_path
    session['new_path'] = new_path
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run()
