import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy import optimize
from scipy.stats import ttest_ind
import statsmodels.api as sm
from statsmodels.formula.api import ols


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








