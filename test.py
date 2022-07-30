from urllib.request import urlopen
url = "https://github.com/nytimes/covid-19-data/blob/master/us-counties-2020.csv"
site = urlopen(url)
meta = site.info()  # get header of the http request
filetype = meta["content-type"]
filetype = filetype.split(';', 1)[0]
print(filetype.rsplit('/', 1)[-1])
# if meta["content-type"] in image_formats:  # check if the content-type is a image
#     print("it is an image")
import pandas as pd
df = pd.read_csv('https://people.sc.fsu.edu/~jburkardt/data/csv/cities.csv')