
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy import optimize
from scipy.stats import ttest_ind
import statsmodels.api as sm
from statsmodels.formula.api import ols

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

def polyfit(x, y, degree):
    results = {}

    coeffs = np.polyfit(x, y, degree)

     # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot
    return results

def polyfunc(x, a, b, c):
    y = a*x*x + b*x+c
    return y

def expofunc(x, a, b):
    y = a*np.exp(b*x)
    return y


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


temperature_df = pd.read_csv("temperature_data.csv")
covid_df = pd.read_csv("covid_2021_losangeles_data.csv")

covid_df2 = covid_df.loc[:,["cases"]]
f_df = pd.concat([temperature_df.loc[:,["Date time","Minimum Temperature","Maximum Temperature","Temperature"]],\
    covid_df2,temperature_df.loc[:,["Address"]]], axis=1)
f_df = f_df.set_index("Date time")

# first figure scatter plot, Cases vs Temperature
fig, (axis1,axis2) = plt.subplots(1,2)

axis1.plot(f_df["Minimum Temperature"],f_df[["cases"]],"bo")
axis1.set(xlabel = "Minimum Temperature F", ylabel = "Daily New Cases")
axis2.plot(f_df["Maximum Temperature"],f_df[["cases"]],"bo")
axis2.set(xlabel = "Maximum Temperature F", ylabel = "Daily New Cases")

fig.suptitle('Daily New Cases vs Daily Temperature')
plt.tight_layout()
plt.show()
plt.savefig("cases_vs_temperature.png")
# nonlinear relationship, some outliers

# second figure hitogram Daily cases
plt.hist(f_df[["cases"]],bins = 20)
plt.xlabel("Daily New Cases")
plt.ylabel("Frequency")
plt.title("Daily New Cases Historgram")
plt.savefig("cases_histogram.png")
#plt.close()
# samples n = 365
# The heights of the bars tell us how many data points are in each bin.

# third figure boxplot
plt.figure(figsize=(7, 5))
sns.boxplot(data=f_df["cases"])
plt.title("Covid Daily New Cases Boxplot")
plt.savefig("Covid Daily New Cases Boxplot.png")
# data lower than 50% of the whole cases numbers are close to the minimum
# many outliers

# correlation
print(np.corrcoef(f_df["Maximum Temperature"], f_df["cases"]))
print(np.corrcoef(f_df["Minimum Temperature"], f_df["cases"]))
# this is another way to get the corrlation, the results are the same
stats.pearsonr(f_df["Maximum Temperature"], f_df["cases"],alternative='two-sided')
stats.pearsonr(f_df["Minimum Temperature"], f_df["cases"],alternative='two-sided')

# try quadratic for Maximum Temperature
# https://pythonnumericalmethods.berkeley.edu/notebooks/chapter16.05-Least-Square-Regression-for-Nonlinear-Functions.html
x = f_df.reset_index()["Maximum Temperature"]
y = f_df.reset_index()["cases"]

alpha, beta, intercept = optimize.curve_fit(polyfunc, xdata = x, ydata = y)[0]
print(f'alpha={alpha}, beta={beta}, intercept = {intercept}')
# alpha=18.135717611263434, beta=-2800.0103960413016, intercept = 108894.94386337163
    
print(polyfit(x,y,2))
#{'polynomial': [18.1357162379303, -2800.0101940512172, 108894.93653946006], 'determination': 0.19431642161659518}
# determination is the R-squared

# Let's have a look of the data
plt.figure(figsize = (10,8))
plt.plot(x, y, 'b.')
plt.plot(x, alpha*x*x + beta*x + intercept, 'r')
plt.xlabel('Maximum Temperature F')
plt.ylabel("Daily New Cases")
plt.title("Polynomial Regression for Daily New Cases vs Maximum Temperature")
plt.savefig("Polynomial Regression for Daily New Cases vs Maximum Temperature.png")
plt.show()

# try exponential for Maximum Temperature

alpha, beta = optimize.curve_fit(expofunc, xdata = x, ydata = y)[0]
print(f'alpha={alpha}, beta={beta}')
# alpha=8.296280126101653e-14, beta=1.0000000064113848

# Let's have a look of the data
plt.figure(figsize = (10,8))
plt.plot(x, y, 'b.')
plt.plot(x, alpha*np.exp(beta*x), 'r')
plt.xlabel('Maximum Temperature F')
plt.ylabel("Daily New Cases")
plt.title("Exponential Regression for Daily New Cases vs Maximum Temperature")
plt.savefig("Exponential Regression for Daily New Cases vs Maximum Temperature.png")
plt.show()

# try quadratic for Minimum Temperature
x = f_df.reset_index()["Minimum Temperature"]
y = f_df.reset_index()["cases"]

alpha, beta, intercept = optimize.curve_fit(polyfunc, xdata = x, ydata = y)[0]
print(f'alpha={alpha}, beta={beta}, intercept = {intercept}')
# alpha=23.68981666653756, beta=-2880.2125452059577, intercept = 88502.63335505218
# Let's have a look of the data
plt.figure(figsize = (10,8))
plt.plot(x, y, 'b.')
plt.plot(x, alpha*x*x + beta*x + intercept, 'r')
plt.xlabel('Minimum Temperature F')
plt.ylabel("Daily New Cases")
plt.title("Polynomial Regression for Daily New Cases vs Minimum Temperature")
plt.savefig("Polynomial Regression for Daily New Cases vs Minimum Temperature.png")
plt.show()

print(polyfit(x,y,2))
# {'polynomial': [23.68981727894456, -2880.212615210808, 88502.63532531407], 'determination': 0.1810859969573649}
# Maximum temperature R2: 0.19431642161659518
# Minimum temperature R2: 0.1810859969573649

# t test
ttest,pval = ttest_ind(f_df["Maximum Temperature"],f_df["cases"],equal_var = False, alternative= "two-sided")
print(f"The statistic t value is {ttest} and the p-value is {pval}")
#The statistic t value is -10.007062739291262 and the p-value is 5.4953356644709336e-21

ttest,pval = ttest_ind(f_df["Minimum Temperature"],f_df["cases"],equal_var = False, alternative= "two-sided")
print(f"The statistic t value is {ttest} and the p-value is {pval}")
# The statistic t value is -10.074685356004688 and the p-value is 3.21249495126667e-21

# factorial ANOVA https://www.reneshbedre.com/blog/anova.html
f2_df = pd.concat([temperature_df.loc[:,["Date time","Minimum Temperature","Dew Point","Relative Humidity"]],\
    covid_df2], axis=1)

f2_df = f2_df.set_index("Date time")
f2_df = f2_df.rename(columns = {"Minimum Temperature": "Minimum_Temperature", "Dew Point":"Dew_Point","Relative Humidity":"Relative_Humidity"})
f2_df.head()

# form correlation matrix
matrix = f2_df.corr()
print("Correlation matrix is : ")
print(matrix)
'''
Correlation matrix is : 
                     Minimum_Temperature  Dew_Point  Relative_Humidity  
Minimum_Temperature             1.000000   0.787565           0.209791   
Dew_Point                       0.787565   1.000000           0.724369   
Relative_Humidity               0.209791   0.724369           1.000000   
cases                          -0.327615  -0.168901           0.092554  

minimum temperature and dew point highly correlated
minimum temperature and relative humidity weakly correlated
dew point and relative humidity higly correlated
can add relative humidity into the model
'''
'''
f2_df = f2_df.loc[:,["Minimum_Temperature","Relative_Humidity","cases"]]

data=f2_df.iloc[0:31]

model = ols('cases ~ C(Minimum_Temperature) + C(Relative_Humidity)+ C(Minimum_Temperature):C(Relative_Humidity)', data).fit()
print(model.summary())
'''




