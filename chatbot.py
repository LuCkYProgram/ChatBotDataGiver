import os
from wolframalpha import Client
from decouple import config
import pyttsx3

API_KEY = config('KEY')
client = Client(API_KEY) # API Key

#Text to Speech
engine = pyttsx3.init()
engine.setProperty("rate", 200)
engine.say("What is your question?")
engine.runAndWait()

#Search Engine Section
question = input("What is your question?")
response = client.query(question)
pod = next(response.results)
answer = pod.text

#Text to Speech
print("Your question: {}".format(question))
engine.say(answer)
engine.runAndWait()
print("Answer: {}".format(answer))
