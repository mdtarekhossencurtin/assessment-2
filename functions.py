import requests
import time
import os
from dotenv import load_dotenv
import json
from collections import defaultdict
import string
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f"{working_dir}/config.json"))

def get_weather_data(location, forecast_days=5):
    WEATHER_URL = config_data["WEATHER_URL"]
    WEATHER_API = config_data["WEATHER_API"]
    WEATHER_AP = config_data["WEATHER_AP"]

    url = WEATHER_URL + location  + WEATHER_API 
    response=requests.get(url)
    get_response=response.json()

    latitude=get_response['coord']['lat']
    longitude = get_response['coord']['lon']

    new_endpoint = f"https://api.openweathermap.org/data/2.5/forecast?lat={latitude}&lon={longitude}&appid={WEATHER_AP}&units=metric"

    forecast_response = requests.get(new_endpoint)
    forecast_data = forecast_response.json()

    
    daily_data = defaultdict(lambda: {"temp": [], "humidity": [], "wind": [],"temp_min":[],"temp_max":[]})

    data = []

    for item in forecast_data["list"]:
        dt_txt = item["dt_txt"]               
        date_str = dt_txt.split(" ")[0]
        time_str = dt_txt.split(" ")[1]


        daily_data[date_str]["temp"].append(item["main"]["temp"])
        daily_data[date_str]["humidity"].append(item["main"]["humidity"])
        daily_data[date_str]["wind"].append(item["wind"]["speed"])
        daily_data[date_str]["temp_min"].append(item["main"]["temp_min"])
        daily_data[date_str]["temp_max"].append(item["main"]["temp_max"])

        row = {"Location":location,"Date": date_str, "Time": time_str,
               "Temperature": item["main"]["temp"], 
               "Humidity":item["main"]["humidity"],
               "Wind_Speed":item["wind"]["speed"],"Longitude":longitude,"Langitude":latitude}
        
        data.append(row)
    
    df = pd.DataFrame(data)

    weather_data = {"location": location}
    daily_summary = {}
    for date, values in daily_data.items():
        avg_temp = sum(values["temp"]) / len(values["temp"])
        avg_humidity = sum(values["humidity"]) / len(values["humidity"])
        avg_wind = sum(values["wind"]) / len(values["wind"])
        avg_min_weather = sum(values["temp_min"]) / len(values["temp_min"])
        avg_max_weather = sum(values["temp_max"]) / len(values["temp_max"])
    
        daily_summary[date] = {
        "temp": round(avg_temp, 2),
        "humidity": round(avg_humidity, 2),
        "wind_speed": round(avg_wind, 2),
        "temperate_min": round(avg_min_weather,2),
        "temperate_max": round(avg_max_weather,2)
        }
    weather_data["forecast"] = daily_summary
    
    return df,weather_data

def parse_weather_question(question):
    australian_cities = [
    "Sydney",
    "Melbourne",
    "Brisbane",
    "Perth",
    "Adelaide",
    "Canberra",
    "Hobart",
    "Darwin",
    "Gold Coast",
    "Newcastle",
    "Wollongong",
    "Cairns",
    "Toowoomba",
    "Sunshine Coast",
    "Geelong",
    "Townsville",
    "Ipswich",
    "Ballarat",
    "Bendigo",
    "Mackay",
    "Rockhampton",
    "Bundaberg",
    "Gladstone",
    "Launceston",
    "Burnie",
    "Devonport"
    ]
    time = ["tomorrow","day after tommorow","in the 2nd day","in the 4th day","in the 5th day","Tomorrow"]
    attribute = ["temp","temperature","humidity","wind speed"]
    action = ['hiking','driving','swimming','going outside']
    question_dict = {"Location":["Perth"],"Time":"today","Attribute":"temperature",'action':""}

    translator = str.maketrans('', '', string.punctuation)
    question = question.translate(translator)

    question = question.split()

    for i in question:
        if i in australian_cities:
            question_dict["Location"].append(i)
        elif i in attribute:
            question_dict["Attribute"] = i
        elif i in time:
            question_dict["Time"] = i
        elif i in action:
            question_dict["action"] = i
    return question_dict


def generate_weather_response(parsed_question, weather_data):
    GROQ_API_KEY = config_data["GROQ_API_KEY"]
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    Expert_BOT_TEMPLATE = """
    You're a weather anylyst.
    Now your work is to focus on {parsed_question} and give answer based on {weather_data}.
    Energetic, exciting tone and give detailed analysis and recommendation, and write it within 300 words. 
    """
    prompt = ChatPromptTemplate.from_template(Expert_BOT_TEMPLATE)
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.8)
    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({"parsed_question":parsed_question,"weather_data": weather_data})
    
    return result

def create_temperature_visualisation(df, output_type='display'):
    if output_type == 'display':

        folder_path = os.path.join("static", "plots")
        file_name = "scatter_plot.png"
        os.makedirs(folder_path, exist_ok=True)
        full_path = os.path.join(folder_path, file_name)
        plt.figure(figsize=(8, 3)) 
        sns.scatterplot(data=df, x='Time', y='Temperature', hue='Date')
        plt.title("Temperature By Time And Temperature")      
        plt.legend(fontsize=3)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        plt.close()

        new_file_name = "Temperature.png"
        new_path = os.path.join(folder_path, new_file_name)
        plt.figure(figsize=(8, 3))
        sns.barplot(data=df, x='Location', y='Humidity', estimator=np.max, errorbar=None)
        plt.title("Maximum Humidity by Location")
        plt.savefig(new_path, dpi=300, bbox_inches='tight')
        plt.close()
        full_path = f"/static/plots/{file_name}"
        new_path = f"/static/plots/{new_file_name}"
        return full_path,new_path