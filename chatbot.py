import os
from wolframalpha import Client
from decouple import config

API_KEY = config('KEY')

client = Client(API_KEY) # API Key

question = input("What is your question?")

response = client.query(question)

pod = next(response.results)

answer = pod.text

print("Your question: {}".format(question))
print("Answer: {}".format(answer))
