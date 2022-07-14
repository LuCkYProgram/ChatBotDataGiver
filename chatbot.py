from ctypes.wintypes import PDWORD
from wolframalpha import Client

client = Client('PQKR53-4VJTP72EUK') # API Key

question = input("What is your question?")

response = client.query(question)

pod = next(response.results)

answer = pod.text

print("Your question: {}".format(question))
print("Answer: {}".format(answer))
