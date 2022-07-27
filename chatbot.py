from logging.config import listen
import os
from decouple import config

import pyttsx3
import speech_recognition

from wolframalpha import Client
from googlesearch import search
import requests
from bs4 import BeautifulSoup

API_KEY = config('KEY')
client = Client(API_KEY) # API Key

recognizer = speech_recognition.Recognizer()
microphone = speech_recognition.Microphone()

#Text to Speech
engine = pyttsx3.init()
voices = engine.getProperty("voices")
voice_id_to_use = voices[1].id
engine.setProperty("voice", voice_id_to_use)
engine.setProperty("rate", 200)

engine.say("What would you like to do? Say or type MATH or GOOGLE or DATASET.")
engine.runAndWait()
with microphone as source:
    recognizer.adjust_for_ambient_noise(source)
with microphone as source:
    print("Listening...")
    audio = recognizer.listen(source)
choice = recognizer.recognize_google(audio)
print("You said: " + choice)

#Search Engine Section
#question = input("What is your question?")

# if question == "covid-19":
#     answer = "Here is the most updated website by John Hopkins."
#     print(answer)
#     engine.say(answer)
#     engine.runAndWait()
#     print("https://coronavirus.jhu.edu/map.html")

def data_google(data):
    dataset = data
    dataset_clean = dataset.replace(' ', '+').lower()
    link = f'https://datasetsearch.research.google.com/search?query={dataset_clean}'

    req = requests.get(link)
    soup = BeautifulSoup(req.content, 'html.parser')

    name = soup.find_all('h1',  class_="iKH1Bc")
    site = soup.find_all('li', class_="iW1HZe")
    print('\n\n--------------------DATASETS FOUND IN GOOGLE DATA SEARCH\n\n')
    print(f'Link to access this and others datasets:{link}')
    for i in range(5):
        name_clean = str(name[i]).split('">')[1].split('</')[0]
        site_clean = str(site[i]).split('">')[1].split('<')[0]
    print(f'\n{name_clean}\n Dataset source: {site_clean}\n')

def listening_replying_function(to_ask):
    engine.say(to_ask)
    engine.runAndWait()
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
    with microphone as source:
        print("Listening...")
        audio = recognizer.listen(source)
    result = recognizer.recognize_google(audio)
    engine.say("You said: " + result)
    engine.runAndWait()
    return result

if choice == "Google":
    query = listening_replying_function("What is your query?")
    engine.say("Here is your top 10 results of your query based by Google.")
    engine.runAndWait()
    print("Here is your top 10 results of your query based by Google.")
    for j in search(query, tld="co.in", num=10, stop=10, pause=2):
        print(j)
elif choice == "math":
    question = listening_replying_function("What is your math question?")
    response = client.query(question)
    pod = next(response.results)
    answer = pod.text
    print("Your question: {}".format(question))
    engine.say("The answer is " + answer)
    engine.runAndWait()
    print("Answer: {}".format(answer))
elif choice == "dataset":
    key_word = listening_replying_function("What dataset do you want to find?")
    data_google(key_word)


