import os
from wolframalpha import Client
from decouple import config
import pyttsx3
import speech_recognition
from googlesearch import search

API_KEY = config('KEY')
client = Client(API_KEY) # API Key

#Text to Speech
engine = pyttsx3.init()
voices = engine.getProperty("voices")
voice_id_to_use = voices[1].id
engine.setProperty("voice", voice_id_to_use)
engine.setProperty("rate", 200)

engine.say("What would you like to do? Say or type MATH or GOOGLE.")
engine.runAndWait()

#Speech To Text
recognizer = speech_recognition.Recognizer()
microphone = speech_recognition.Microphone()
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

if choice == "Google":
    engine.say("What is your query?")
    engine.runAndWait()
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
    with microphone as source:
        print("Listening...")
        audio = recognizer.listen(source)
    query = recognizer.recognize_google(audio)
    engine.say("You said: " + query)
    engine.runAndWait()
    engine.say("Here is your top 10 results of your query based by Google.")
    engine.runAndWait()
    print("Here is your top 10 results of your query based by Google.")
    for j in search(query, tld="co.in", num=10, stop=10, pause=2):
        print(j)
elif choice == "math":
    engine.say("What is your question?")
    engine.runAndWait()
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
    with microphone as source:
        print("Listening...")
        audio = recognizer.listen(source)
    question = recognizer.recognize_google(audio)
    response = client.query(question)
    pod = next(response.results)
    answer = pod.text
    #Text to Speech
    print("Your question: {}".format(question))
    engine.say(answer)
    engine.runAndWait()
    print("Answer: {}".format(answer))


