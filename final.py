'''
import json
import urllib.request

#import pdb; pdb.set_trace()
'''

import requests
import pandas as pd

def get_termperature_data(start_date, end_date):
	'''
	This function is going to get the temperature data using API.
	Since the time period for API is a month, I need to call this function 12 times to get 2021 data.

	Args:
		start_date: the start date of the time period
		end_date: the ending date of the time period

	Returns:
		data: raw data from the api
	'''

	url = "https://visual-crossing-weather.p.rapidapi.com/history"

	querystring = {
				"startDateTime":f"{start_date}",
				"aggregateHours":"24",
				"location":"Los Angeles,CA,USA",
				"endDateTime":f"{end_date}",
				"unitGroup":"us",
				"contentType":"csv",
				"shortColumnNames":"0"
	}

	headers = {
		"X-RapidAPI-Key": "be010c0552msh31daec782930918p184eb5jsnf3d4c0c3ee26",
		"X-RapidAPI-Host": "visual-crossing-weather.p.rapidapi.com"
	}

	response = requests.request("GET", url, headers=headers, params=querystring)
	
	return response


temperature_date = (
		("2021-01-01T00:00:00","2021-01-31T00:00:00"),
		("2021-02-01T00:00:00","2021-02-28T00:00:00"),
		("2021-03-01T00:00:00","2021-03-31T00:00:00"),
		("2021-04-01T00:00:00","2021-04-30T00:00:00"),
		("2021-05-01T00:00:00","2021-05-31T00:00:00"),
		("2021-06-01T00:00:00","2021-06-30T00:00:00"),		
		("2021-07-01T00:00:00","2021-07-31T00:00:00"),
		("2021-08-01T00:00:00","2021-08-31T00:00:00"),
		("2021-09-01T00:00:00","2021-09-30T00:00:00"),		
		("2021-10-01T00:00:00","2021-10-31T00:00:00"),
		("2021-11-01T00:00:00","2021-11-30T00:00:00"),
		("2021-12-01T00:00:00","2021-12-31T00:00:00"),		
	)
'''
with open("temperature_data.csv", "w") as fout:
	date = temperature_date[0]
	response = get_termperature_data(date[0],date[1])
	fout.write(response.text)
#fout =open("temperature_data.csv", "w")
#fout.close()
for i in range(1,len(temperature_date)):
	date = temperature_date[i]
	response = get_termperature_data(date[0],date[1])
	with open("temperature_data.csv", "a") as fout:
		fout.write(response.text)

temperature_df = pd.read_csv("temperature_data.csv")
print(temperature_df)
'''
for i in range(len(temperature_date)):
	date = temperature_date[i]
	response = get_termperature_data(date[0],date[1])

	with open(f"{i+1}_temperature_data.csv", "w") as fout:
		fout.write(response.text)
	#temperature_df = pd.read_csv("temperature_data.csv")

#read monthly temperature data
temperature_df1 = pd.read_csv("1_temperature_data.csv")
temperature_df2 = pd.read_csv("2_temperature_data.csv")
temperature_df3 = pd.read_csv("3_temperature_data.csv")
temperature_df4 = pd.read_csv("4_temperature_data.csv")
temperature_df5 = pd.read_csv("5_temperature_data.csv")
temperature_df6 = pd.read_csv("6_temperature_data.csv")
temperature_df7 = pd.read_csv("7_temperature_data.csv")
temperature_df8 = pd.read_csv("8_temperature_data.csv")
temperature_df9 = pd.read_csv("9_temperature_data.csv")
temperature_df10 = pd.read_csv("10_temperature_data.csv")
temperature_df11 = pd.read_csv("11_temperature_data.csv")
temperature_df12 = pd.read_csv("12_temperature_data.csv")

#append rows to get the 2021 temperature data from 12 monthly dataframes
temperature_df = pd.concat(
	[temperature_df1,temperature_df2,temperature_df3,temperature_df4,
	temperature_df5,temperature_df6,temperature_df7,temperature_df8,
	temperature_df9,temperature_df10,temperature_df11,temperature_df12]
)

#save all the temperature data for 2021 into a csv file
temperature_df.to_csv("temperature_data.csv",index = False)

#collect covid data
covid_df = pd.read_csv("covid19cases_test.csv")
#covid_df = covid_df[covid_df["area"] == "Los Angeles"]
covid_df = covid_df.loc[19848:20212,:]
# save covid data for los angeles in 2021 into a csv file
covid_df.to_csv("covid_2021_losangeles_data.csv",index = False)



