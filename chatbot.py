from models import *
#config modules
from logging.config import listen
from operator import is_
import os
from decouple import config
#speech to text to speech modules
import pyttsx3
import speech_recognition
#searches modules
from wolframalpha import Client
from googlesearch import search
import requests
from bs4 import BeautifulSoup
#data science / modeling modules
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np



from urllib.request import urlopen

def file_ext(url):
    site = urlopen(url)
    meta = site.info()  # get header of the http request
    filetype = meta["content-type"]
    filetype = filetype.split(';', 1)[0]
    return filetype.rsplit('/', 1)[-1]

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

def google_search(data):
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

def awesome_search(data):
    name_selec = []
    link_selec = []

    link = 'https://github.com/awesomedata/awesome-public-datasets'
    req = requests.get(link)
    soup = BeautifulSoup(req.content, 'html.parser')
    html = soup.find_all('a', rel="nofollow")
    print('\n\n-------------------- DATASETS FOUND IN AWESOME PUBLIC DATASETS\n\n')

    for i in range(len(html))[8:]:
        name = (str(html[i]).strip('</a>').split('w">')[1]).lower()
        link = str(html[i]).split('f="')[1].split('" r')[0]
        if (data in name):
            name_selec.append(name)
            link_selec.append(link)
        else:
            name_selec += ''
            link_selec += ''
    for j in range(len(name_selec)):
        if len(name_selec) < 1:
            print('Datasets not found')
        else:
            print(f'\n{name_selec[j]}\n{link_selec[j]}\n')

def uci_search(data):
    dataset = data
    dataset_clean = dataset.replace(' ', '+').lower()

    link = f'https://archive.ics.uci.edu/ml/datasets/{dataset_clean}'
    req = requests.get(link)
    soup = BeautifulSoup(req.content, 'html.parser')

    dataset_html = soup.find_all('span', class_='heading')
    table = soup.find_all('td')
    warning = soup.find_all('p')
    print('\n\n-------------------- DATASETS FOUND IN UCI\n\n')
    try:
        name = str(dataset_html).split('b>')[1].split('</')[0]
        charac = str(table).split('"normal">')[46].split('</p>')[0]
    except IndexError:
        name = ''
        charac = ''
        link = ''
        print('')

    print(f'\n{name} ({link})\n{charac}')

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


def stat_analysis(section,df):
    var = df[section]

    # Get statistics
    min_val = var.min()
    max_val = var.max()
    mean_val = var.mean()
    med_val = var.median()
    mod_val = var.mode()[0]

    print('Minimum:{:.2f}\nMean:{:.2f}\nMedian:{:.2f}\nMode:{:.2f}\nMaximum:{:.2f}\n'.format(min_val,
                                                                                            mean_val,
                                                                                            med_val,
                                                                                            mod_val,
                                                                                            max_val))

    # Create a Figure
    fig = plt.figure(figsize=(10,4))

    # Plot a histogram
    plt.hist(var)

    # Add lines for the statistics
    plt.axvline(x=min_val, color = 'gray', linestyle='dashed', linewidth = 2)
    plt.axvline(x=mean_val, color = 'cyan', linestyle='dashed', linewidth = 2)
    plt.axvline(x=med_val, color = 'red', linestyle='dashed', linewidth = 2)
    plt.axvline(x=mod_val, color = 'yellow', linestyle='dashed', linewidth = 2)
    plt.axvline(x=max_val, color = 'gray', linestyle='dashed', linewidth = 2)

    # Add titles and labels
    plt.title('Data Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # Show the figure
    fig.show()
    
def split_data(df):
    df = df.dropna()
    columns = list(df)
    for i, s in enumerate(columns):
        columns[i] = columns[i].strip()
        columns[i] = columns[i].strip('"')
    label = input("Select model label: ")
    label_index = columns.index(label)
    features = list(df)
    labels = features.pop(label_index)
    from sklearn.model_selection import train_test_split
    X, y = df[features].values, df[labels].values
    print('Features:',X[:10], '\nLabels:', y[:10], sep='\n')
    # Split data 70%-30% into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    print ('Training Set: %d rows\nTest Set: %d rows' % (X_train.shape[0], X_test.shape[0]))
    return(X_train, X_test, y_train, y_test, features, labels)

def feature_distributions(df,features,labels):
    for col in features:
        df.boxplot(column=col, by=labels, figsize=(6,6))
        plt.title(col)
    plt.show()

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
    google_search(key_word)
    uci_search(key_word)
    # awesome_search(key_word)
    engine.say("Please copy and paste the link needing to be used format into a data set.")
    engine.runAndWait()
    data_url = input("URL: ")
    file_type = file_ext(data_url)
    try:
        df = pd.read_csv(data_url)
    except:
        print("This program does not support " + file_type + " file extension type.")
        exit()
    df.head()
    column = input("column name: ")
    stat_analysis(column,df)
    model_type = input("Is this a Regression, Classification, Clustering, or Deep Learning Model?")
    if model_type == "Regression":
        X_train, X_test, y_train, y_test = split_data(df)
        model = train_regression(X_train, y_train)
        evaluation_of_regression(model, X_test, y_test)
    elif model_type == "Classification":
        bin_or_multi = input("Binary or Multi?")
        if bin_or_multi == "Binary":
            X_train, X_test, y_train, y_test = split_data(df)
            model = train_binary_classification(X_train, y_train)
            evaluation_of_binary_classification(model, X_test, y_test)
        else:
            X_train, X_test, y_train, y_test = split_data(df)
            model = train_multi_classification(X_train, y_train)
            evaluation_of_multi_classification(model, X_test, y_test)
            

