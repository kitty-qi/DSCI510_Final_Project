'''
import json
import urllib.request

#import pdb; pdb.set_trace()
'''

import requests
import pandas as pd

url = "https://visual-crossing-weather.p.rapidapi.com/history"

querystring = {
			"startDateTime":"2021-01-01T00:00:00",
			"aggregateHours":"24",
			"location":"Los Angeles,CA,USA",
			"endDateTime":"2021-01-31T00:00:00",
			"unitGroup":"us",
			"dayStartTime":"8:00:00",
			"contentType":"csv",
			"dayEndTime":"17:00:00",
			"shortColumnNames":"0"
}

headers = {
	"X-RapidAPI-Key": "be010c0552msh31daec782930918p184eb5jsnf3d4c0c3ee26",
	"X-RapidAPI-Host": "visual-crossing-weather.p.rapidapi.com"
}

response = requests.request("GET", url, headers=headers, params=querystring)
with open("temperature_data.csv", "w") as fout:
	fout.write(response.text)
temperature_df = pd.read_csv("temperature_data.csv")
print(temperature_df)



