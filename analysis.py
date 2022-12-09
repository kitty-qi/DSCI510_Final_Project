import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy import optimize
#%matplotlib inline


covid_df = pd.read_csv("covid_2021_losangeles_data.csv")
temperature_df = pd.read_csv("temperature_data.csv")

#temperature_df2 = temperature_df.loc[:,["Date time","Minimum Temperature","Maximum Temperature","Temperature","Address"]]
covid_df2 = covid_df.loc[:,["cases"]]

f_df = pd.concat([temperature_df.loc[:,["Date time","Minimum Temperature","Maximum Temperature","Temperature"]],\
	covid_df2,temperature_df.loc[:,["Address"]]], axis=1)

f_df = f_df.set_index("Date time")

# Daily new cases vs daily temperature figure
fig, (axis1,axis2) = plt.subplots(1,2)

axis1.plot(f_df["Minimum Temperature"],f_df[["cases"]],"bo")

axis1.set(xlabel = "Minimum Temperature", ylabel = "Daily New Cases")
axis2.plot(f_df["Maximum Temperature"],f_df[["cases"]],"bo")
axis2.set(xlabel = "Maximum Temperature", ylabel = "Daily New Cases")

fig.suptitle('Daily New Cases vs Daily Temperature')
plt.tight_layout()
plt.show()
plt.savefig("cases_vs_temperature.png")


# nonlinear relationship, some outliers

f_df["cases"].describe()

plt.hist(f_df[["cases"]],bins = 20)
plt.xlabel("Daily New Cases")
plt.ylabel("Frequency")
plt.title("Daily New Cases Historgram")
plt.savefig("cases_histogram.png")
#plt.close()
# samples n = 365
# The heights of the bars tell us how many data points are in each bin.

plt.figure(figsize=(7, 5))
sns.boxplot(data=f_df["cases"])
plt.title("Covid Daily New Cases Boxplot")
plt.savefig("Covid Daily New Cases Boxplot.png")
# data lower than 50% of the whole cases numbers are close to the min
# many outliers

stats.pearsonr(f_df["Maximum Temperature"], f_df["cases"],alternative='two-sided')
stats.pearsonr(f_df["Minimum Temperature"], f_df["cases"],alternative='two-sided')


# try quadratic , exponential for Maximum Temperature
x = f_df.reset_index()["Maximum Temperature"]
y = f_df.reset_index()["cases"]

# let's define the function form
def func(x, a, b, c):
    y = a*x*x + b*x+c
    return y

alpha, beta, intercept = optimize.curve_fit(func, xdata = x, ydata = y)[0]
print(f'alpha={alpha}, beta={beta}, intercept = {intercept}')

# Let's have a look of the data
plt.figure(figsize = (10,8))
plt.plot(x, y, 'b.')
plt.plot(x, alpha*x*x + beta*x + intercept, 'r')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# let's define the function form
def func(x, a, b):
    y = a*np.exp(b*x)
    return y

alpha, beta = optimize.curve_fit(func, xdata = x, ydata = y)[0]
print(f'alpha={alpha}, beta={beta}')

# Let's have a look of the data
plt.figure(figsize = (10,8))
plt.plot(x, y, 'b.')
plt.plot(x, alpha*np.exp(beta*x), 'r')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# try quadratic , exponential for Minimum Temperature
x = f_df.reset_index()["Minimum Temperature"]
y = f_df.reset_index()["cases"]

# let's define the function form
def func(x, a, b, c):
    y = a*x*x + b*x+c
    return y

alpha, beta, intercept = optimize.curve_fit(func, xdata = x, ydata = y)[0]
print(f'alpha={alpha}, beta={beta}, intercept = {intercept}')

# Let's have a look of the data
plt.figure(figsize = (10,8))
plt.plot(x, y, 'b.')
plt.plot(x, alpha*x*x + beta*x + intercept, 'r')
plt.xlabel('x')
plt.ylabel('y')
plt.show()



'''
print(f_df.head())
print(f_df.tail())
print(f_df.describe())

'''