#!/usr/bin/env python
# coding: utf-8

# $$\large \color{green}{\textbf{The Daily Stock Price Analysis and Forecasting with the ARIMA Model}}$$ 
# 
# $$\large \color{blue}{\textbf{Phuong Van Nguyen}}$$
# $$\small \color{red}{\textbf{ phuong.nguyen@summer.barcelonagse.eu}}$$
# 
# 
# This Machine Learning program was written by Phuong V. Nguyen, based on the $\textbf{Anacoda 1.9.7}$ and $\textbf{Python 3.7}$.
# 
# 
# $$\underline{\textbf{Main Contents}}$$
# 
# $\text{1. Main Job:}$ This program is to analyze and predict the close price of Vingroup Stock.
# 
# $\text{2. Dataset:}$ 
# https://www.vndirect.com.vn/portal/thong-ke-thi-truong-chung-khoan/lich-su-gia.shtml

# # Preparing Problem
# 
# ##  Loading Libraries
# 
# 

# In[2]:


import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima_model import ARIMA
#import pmdarima as pm
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'


# ## Defining some varibales for printing the result

# In[3]:


Purple= '\033[95m'
Cyan= '\033[96m'
Darkcyan= '\033[36m'
Blue = '\033[94m'
Green = '\033[92m'
Yellow = '\033[93m'
Red = '\033[91m'
Bold = "\033[1m"
Reset = "\033[0;0m"
Underline= '\033[4m'
End = '\033[0m'


# ##  Loading Dataset

# In[ ]:


df = pd.read_excel("VICprice.xlsx")
df.head(5)


# # Data Process and Exploration
# 
# ## Data exploration
# 
# ### Statisical descriptive
# 
# #### Checking the data size and format

# In[8]:


df.shape


# In[9]:


df.dtypes


# #### Selecting a specific dataframe

# In[10]:


closePrice = df[['DATE','CLOSE']]


# In[11]:


closePrice.head(5)


# In[442]:


closePrice['DATE'].min(), closePrice['DATE'].max()


# #### Checking whether data has missing values

# In[12]:


closePrice.isnull().sum()


# No missing value in the dataframe "closePrice"

# #### Checking the statistical basic

# In[13]:


closePrice.CLOSE.describe()


# #### Checking the Data Type

# In[14]:


closePrice.dtypes


# ### Visualization
# 
# #### Line plot

# In[16]:


p_vin=closePrice
p_vin = p_vin.set_index('DATE')
p_vin.head()


# In[18]:


sns.set()
fig=plt.figure(figsize=(12,7))
plt.plot(p_vin.CLOSE['2007':'2019'],LineWidth=4)
plt.autoscale(enable=True,axis='both',tight=True)
#plt.grid(linestyle=':',which='both',linewidth=2)
fig.suptitle('The Daily Close Prices of The VINGROUP (VIC)', fontsize=18,fontweight='bold')
plt.title('19/09/2007- 27/12/2019',fontsize=15,fontweight='bold',color='k')
plt.ylabel('1,000 VND',fontsize=17)
plt.xlabel('Source: VNDirect Securities Com.',fontsize=17,fontweight='normal',color='k')


# #### The Box and Whisker Plots
# 
# #####  By Month in Year

# In[40]:


p_vinmy=closePrice[['DATE','CLOSE']]
p_vinmy.head()


# In[41]:


p_vinmy['Month-Year'] = p_vinmy['DATE'].dt.strftime('%b-%y')
p_vinmy.head()


# In[42]:


p_vinmy = p_vinmy.set_index('DATE')
p_vinmy.head()


# In[44]:


sns.set()
fig=plt.figure(figsize=(12,7))
sns.boxplot(x='Month-Year',y='CLOSE',data=p_vinmy['2019':'2019'])
#plt.autoscale(enable=True,axis='both',tight=True)
plt.grid(linestyle=':',which='both',linewidth=2)
fig.suptitle('The Box and Whisker Plots of the VIC Prices by Month in Year', fontsize=18,fontweight='bold')
plt.title('19/09/2007- 27/12/2019',fontsize=15,fontweight='bold',color='k')
plt.ylabel('1,000 VND',fontsize=17)
plt.xlabel('Source: VNDirect Securities Com.',fontsize=17,fontweight='normal',color='k')


# By using the above Boxplot, you can compare the variance of the VIC prices across months

# ##### By Month (Seasonality)

# In[46]:


p_vinm=closePrice[['DATE','CLOSE']]
p_vinm.head()


#  Setting the year group

# In[48]:


p_vinm['Month'] = p_vinm['DATE'].dt.strftime('%b')
p_vinm.head()


#  Indexing Data

# In[49]:


p_vinm=p_vinm.set_index('DATE')
p_vinm.head()


# Creating The Box and Whisker Plots by Month

# In[55]:


sns.set()
fig=plt.figure(figsize=(12,7))
sns.boxplot(x='Month',y='CLOSE',data=p_vinm['2007':'2019'])
#plt.autoscale(enable=True,axis='both',tight=True)
plt.grid(linestyle=':',which='both',linewidth=2)
fig.suptitle('The Box and Whisker Plots of the VIC Prices by Month (Seasonality)', fontsize=18,fontweight='bold')
plt.title('19/09/2007- 27/12/2019',fontsize=15,fontweight='bold',color='k')
plt.ylabel('1,000 VND',fontsize=17)
plt.xlabel('Source: The Daily Close Price-based Calculations ',fontsize=17,fontweight='normal',color='k')


# ##### By Year (Trend)

# In[34]:


p_viny=closePrice[['DATE','CLOSE']]
p_viny.head()


#  Setting the year group

# In[35]:


p_viny['Year'] = p_viny['DATE'].dt.strftime('%Y')
p_viny.head()


# Indexing Data

# In[36]:


p_viny=p_viny.set_index('DATE')
p_viny.head()


# Creating The Box and Whisker Plots by Year

# In[56]:


sns.set()
fig=plt.figure(figsize=(12,7))
sns.boxplot(x='Year',y='CLOSE',data=p_viny['2007':'2019'])
#plt.autoscale(enable=True,axis='both',tight=True)
plt.grid(linestyle=':',which='both',linewidth=2)
fig.suptitle('The Box and Whisker Plots of the VIC Prices by Year (Trend)', fontsize=18,fontweight='bold')
plt.title('19/09/2007- 27/12/2019',fontsize=15,fontweight='bold',color='k')
plt.ylabel('1,000 VND',fontsize=17)
plt.xlabel('Source: VNDirect Securities Com.',fontsize=17,fontweight='normal',color='k')


# By using the Box and Whisker Plot above, one can compare the variance of the VIC prices across years. For example, the VIC prices in the five years of 2009, 2010, 2011, 2012 and 2014 are the most volatile since it started to list in 2007. Oppositely, this stock price was the most stable in 2013 and 2019.

# In[38]:


sns.set()
fig=plt.figure(figsize=(15,7))
sns.boxplot(x='Year',y='CLOSE',data=p_viny['2007':'2019'])
#plt.autoscale(enable=True,axis='both',tight=True)
plt.grid(linestyle=':',which='both',linewidth=2)
fig.suptitle('So Sánh Biến Động Giá Cổ Phiếu VINGROUP Qua Các Năm Kể Từ Khi Niêm Yết', fontsize=18,fontweight='bold')
plt.title('19/09/2007- 27/12/2019',fontsize=15,fontweight='bold',color='k')
plt.ylabel('1,000 VND',fontsize=17)
plt.xlabel('Nguồn: Tính toán dựa trên thống kê giá đóng cửa hàng ngày của Cổ Phiếu VINGROUP',fontsize=17,fontweight='normal',color='k')


# Kể từ cổ phiếu VINGROUP chào sàn ngày 19/09/2007, thì biến động giá của mã cổ phiếu Tập đoàn này, mã giao dịch: $\textbf{VIC}$, có 3 đặc điểm cơ bản sau.
# 
# 1. $\textbf{Biến động nhiều nhất}$ trong 5 năm: 2009,2010, 2011, 2012, và 2014. Nên có thể nói, anh em theo chiến lược lướt sóng thì sẽ chịu $\textbf{rủi ro thua lỗ nhiều nhất}$ trong 5 năm này.
# 
# 2. $\textbf{Ổn định nhất}$ trong hai năm: 2013 và 2019, giá cổ phiếu VIC lại. Ví dụ, giá đóng cửa của mã này trong năm 2019 phổ biến nhất ở mức quanh đâu đó khoảng 116,000 VND/1 Cổ Phiếu. Nên anh em mua cổ phiếu này đầu năm và bán chốt lời cuối năm trên mỗi cổ phiếu thì $\textbf{lời lãi tiêu tết cũng không được là bao}$
# 
# 3. $\textbf{Nhiều bất thường nhất}$ trong năm 2008, và điều này có thể giải thích bởi tác động của Khủng Hoảng Tài chính toàn cầu năm 2008. Những $\textbf{biến động giá bất thường}$ này cũng xảy ra ở các năm 2017 và $\textbf{năm rồi 2019}$.

# ## Data handling
# ### Computing the Return

# In[98]:


closePrice['Return'] = closePrice['CLOSE'].pct_change()
closePrice.head()


# In[99]:


r_vin=closePrice[['DATE','Return']]
r_vin.head()


# In[100]:


r_vin=r_vin.set_index('DATE')


# In[112]:


r_vin.head() 


# In[103]:


sns.set()
fig=plt.figure(figsize=(12,7))
plt.plot(r_vin.Return['2007':'2019'],LineWidth=4)
plt.autoscale(enable=True,axis='both',tight=True)
#plt.grid(linestyle=':',which='both',linewidth=2)
fig.suptitle('The Log Daily Returns of The VINGROUP (VIC) Stock Prices', fontsize=18,fontweight='bold')
plt.title('19/09/2007- 27/12/2019',fontsize=15,fontweight='bold',color='k')
plt.ylabel('Return',fontsize=17)
plt.xlabel('Source: The Daily Close Price-based Calculations',fontsize=17,fontweight='normal',color='k')


# ### Histogram and Density

# In[107]:


sns.set()
fig=plt.figure(figsize=(12,7))
sns.distplot(closePrice[['Return']],bins=150)
plt.autoscale(enable=True,axis='both',tight=True)
#plt.grid(linestyle=':',which='both',linewidth=2)
fig.suptitle('The Histogram of The Daily Returns of The VINGROUP (VIC) Stock Prices', fontsize=18,fontweight='bold')
plt.title('19/09/2007- 27/12/2019',fontsize=15,fontweight='bold',color='k')
plt.ylabel('Percent (%)',fontsize=17)
plt.xlabel('Source: The Daily Close Price-based Calculations',fontsize=17,fontweight='normal',color='k')


# ### The Statistical Basic of the return

# In[116]:


closePrice[['Return']].describe()


# ### Checking right- or left- Skewness

# In[110]:


closePrice[['Return']].skew()


# It implies the daily return has larger towards the left hand side of the distribution or on the left from the peak of the distribution.

# ### Checking thin- or fat distribution

# In[111]:


closePrice[['Return']].kurtosis()


# The distribution of the return shows a positive kurtusis. Accordingly, the peak is stepper and/or the tails are much fatter. It implies an investment on this stock suffers from a low risk since the variance of the return is small. 

# In[148]:


from statsmodels.tsa.stattools import acf 
from statsmodels.tsa.stattools import pacf 
from statsmodels.graphics.tsaplots import plot_acf 
from statsmodels.graphics.tsaplots import plot_pacf 


# In[ ]:


from matplotlib import pyplot


# In[144]:


sns.set()
plot_acf(closePrice[['CLOSE']],lags=365)
plt.autoscale(enable=True,axis='both',tight=True)
plt.title('The Autocorrelation of The Daily Prices of VIC',fontsize=15,fontweight='bold',color='k')
plt.ylabel('Persistence',fontsize=17)
plt.xlabel('A Number of Lags',fontsize=17,fontweight='normal',color='k')


# In[152]:


#sns.set()
plot_pacf(closePrice[['CLOSE']],lags=50)
plt.autoscale(enable=True,axis='both',tight=True)
plt.title('The Partial Autocorrelation of The Daily Prices of VIC',fontsize=15,fontweight='bold',color='k')
plt.ylabel('Persistence',fontsize=17)
plt.xlabel('A Number of Lags',fontsize=17,fontweight='normal',color='k')


# In[160]:


fig=plot_acf(closePrice[['CLOSE']],lags=350)
fig=plot_pacf(closePrice[['CLOSE']],lags=50)


# In[129]:


closePrice[['CLOSE']].head()


# ## Preparing Data for training models
# 
# ### Picking up data

# In[162]:


VIC=closePrice[['DATE','CLOSE']]
VIC.head()


# ### Indexing Data

# In[163]:


VIC =VIC.set_index('DATE')
VIC.head()


# ### Checking the Index of data

# In[164]:


VIC.index


# We can see that it has no frequency (freq=None). This makes sense, since the index was created from a sequence of dates in our CSV file, without explicitly specifying any frequency for the time series.
# 
# We know that our data should be at a specific frequency as a daily basic, we can use the DataFrame’s asfreq() method to assign a frequency. If any date/times are missing in the data, new rows will be added for those date/times, which are either empty (NaN), or filled according to a specified data filling method such as forward filling or interpolation.
# 
# To this end, we use the function "asfreq('B')" to format our series a business day data

# ### Declearing the Frequency of Data as the Business Daily Basic

# In[165]:


VIC=VIC.asfreq('B')#,method='ffill')
VIC=VIC.CLOSE
VIC.head()


# ### Checking if Data contains Missing Values

# In[166]:


VIC.isnull().sum()


# It is $NOT$ good. Please, find an appropriate solution for fulfilling missing data after using the above function $\textbf{asfreq()}$

# # Traing model and performing prediction

# We are going to apply one of the most commonly used method for time-series forecasting, known as ARIMA, which stands for Autoregressive Integrated Moving Average.
# 
# ARIMA models are denoted with the notation $ARIMA(p, d, q)$. These three parameters account for seasonality, trend, and noise in data:

# ## Find the Optimal ARIMA Model

# This step is parameter Selection for our furniture’s sales ARIMA Time Series Model. Our goal here is to use a “grid search” to find the optimal set of parameters that yields the best performance for our model.

# ### Setting a set of hyperparameters

# In[169]:


p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
pdq


# In[355]:


print(Bold + 'A Number of combinations: {}'.format(len(pdq)))


# In[356]:


print(Bold + 'Examples of parameter combinations for ARIMA...' + End)
print('ARIMA: {}'.format(pdq[1]))
print('ARIMA: {}'.format(pdq[2]))
print('ARIMA: {}'.format(pdq[3]))
print('ARIMA: {}'.format(pdq[4]))


# ### Finding the Optimal Set of Hyperparameters 

# In[357]:


AIC=list()
BIC=list()
para=list()
Like=list()
print(Bold + 'Training ARIMA with a Number of Configuration:'+ End)
for param in pdq:
    mod=sm.tsa.statespace.SARIMAX(VIC,order=param,seasonal_order=(0, 0, 0, 0), 
                                               enforce_stationarity=False, enforce_invertibility=False)
    results= mod.fit()
    para.append(param)
    AIC.append(results.aic)
    BIC.append(results.bic)
    Like.append(results.Log)
    print('ARIMA{} - AIC:{} - BIC:{}'.format(param,results.aic, results.bic))
    
print(Bold +'The Optimal Choice Suggestions:'+End)
print('The minimum value of Akaike Information Criterion (AIC):{}'.format(min(AIC)))
print('The minimum value of Bayesian Information Criterion (BIC: {})'.format(min(BIC)))
print(Bold + 'Descending the Values of AIC and BIC:'+End)
ModSelect=pd.DataFrame({'Hyperparameters':para,'AIC':AIC,'BIC':BIC}).sort_values(by=['AIC','BIC'],ascending=False)
ModSelect


# ### Fitting the SARIMA model

# In[512]:


VIC.head()


# In[167]:


mod = sm.tsa.statespace.SARIMAX(VIC, order=(6, 1, 6),enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

print(Bold + 'The estimated ARIMA(6,1,6) Model'+ End)
#print(results.summary().tables[1])
print(results.summary())


# ## Performing Model Diagnostics

# We should always run model diagnostics to investigate any unusual behavior.

# In[511]:


model.plot_diagnostics(figsize=(17,15))
plt.show()


# Our primary concern is to ensure that the residuals of our model are $\textbf{uncorrelated}$ and $\textbf{normally distributed with zero-mean}$. If the seasonal ARIMA model does not satisfy these properties, it should be further improved.
# 
# In this case, our model diagnostics suggests that the model residuals are not normally distributed based on the following:
# 
# 1. The residuals over time (top left plot) seem to display an obvious seasonality (downward trend) and moight not a obvious white noise process.
# 
# 
# 2. In the top right plot, we see that the red KDE line is far from the N(0,1) line (where N(0,1)) is the standard notation for a normal distribution with mean 0 and standard deviation of 1). This is a good indication that the residuals are not normally distributed. 
# 
# 3. The qq-plot on the bottom left shows that the ordered distribution of residuals (blue dots) do not follow the linear trend of the samples taken from a standard normal distribution with N(0, 1). Again, this is a strong indication that the residuals are not normally distributed.
# 
# see more $\text{Q-Q Plot}$ at https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot
# 
# 4. The autocorrelation (i.e. correlogram) plot on the bottom right, which shows that the time series residuals have low correlation with lagged versions of itself, but several espisodes have high correlation with their own lagged values.
# 
# Those observations lead us to conclude that our model produces a satisfactory fit that could help us understand our time series data and forecast future values.
# 
# Although we have a satisfactory fit, some parameters of our seasonal ARIMA model could be changed to improve our model fit. For example, our grid search only considered a restricted set of parameter combinations, so we may find better models if we widened the grid search.
# 
# It is not good enough since our model diagnostics suggests that the model residuals are not near normally distributed.

# ##  Validating Forecasts 

# To help us understand the accuracy of our forecasts, we compare predicted sales to real sales of the time series, and we set forecasts to start at 2017–01–01 to the end of the data.

# ###  In-Sample Forecast
# 
# #### Making insample forecast

# In[513]:


VIC.index


# In[514]:


pred = results.get_prediction(start=pd.to_datetime('2018-06-06'), dynamic=False)


# In[362]:


pred.predicted_mean


# #### Plotting

# In[515]:


pred_ci = pred.conf_int() # Creating the 95% confident interval

ax = VIC['2018-02':].plot(label='observed',LineWidth=4)
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast',LineWidth=4, alpha=.7, figsize=(14, 10))

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('VIC Prices (1,000 VND)')
plt.legend()

plt.show()


# #### Computing MSE

# In statistics, the $\text{Mean Squared Error (MSE)}$ of an estimator measures the average of the squares of the errors — that is, the average squared difference between the estimated values and what is estimated. The MSE is a measure of the quality of an estimator — it is always non-negative, and the smaller the MSE, the closer we are to finding the line of best fit.

# In[365]:


y_forecasted = pred.predicted_mean
y_truth = VIC['2018-06-06':]
se=(y_forecasted - y_truth)**2
mse=se.mean()
print(Bold+'The Mean Squared Error (MSE): {}'.format(round(mse, 2)))


# #### Computing RMSE

# In[366]:


print(Bold + 'The Root Mean Squared Error (RMSE) {}'.format(round(np.sqrt(mse), 2)))


# In[367]:


VIC.describe()


# $\text{Root Mean Square Error (RMSE)}$ tells us that our model was able to forecast the daily price of VIC in the test set within $17,700$ VND of the actual prices. Our daily close price of VIC ranges from around $33,300$ VND to $193,000$ VND. In my opinion, the $ARIMA(6,1,6)$ is a pretty good model so far.

# ### Out-Of-Sample forecasts
# 
# #### Making Out-Of-Sample forecasts

# In[554]:


pred_uc = results.get_forecast(steps=5)
pred_ci = pred_uc.conf_int()
pred_uc.predicted_mean


# #### Visualizing 

# In[555]:


sns.set()
fig=plt.figure(figsize=(12,7))
ax = VIC['2019-12':'2019-12'].plot(label='Giá đã đóng cửa',LineWidth=4)
plt.autoscale(enable=True,axis='both',tight=True)
pred_uc.predicted_mean.plot(ax=ax, label='Giá dự báo',LineWidth=4)
#ax.fill_between(pred_ci.index, 
#                pred_ci.iloc[:, 0],
#                pred_ci.iloc[:, 1], color='k', alpha=.25)
fig.suptitle('Dự Báo Giá Cổ Phiếu VIC Theo Ngày Của Tập Doàn VINGROUP ', fontsize=18,fontweight='bold')
plt.title('',fontsize=18,fontweight='bold')
ax.set_xlabel('Ngày',fontsize=18,fontweight='bold')
ax.set_ylabel('1,000 VND')
plt.grid(linestyle='--',which='both',linewidth=2)
plt.legend()
plt.show()


# In[ ]:


#plt.grid(linestyle=':',which='both',linewidth=2)
fig.suptitle('The Daily Close Prices of The VINGROUP (VIC)', fontsize=18,fontweight='bold')
plt.title('19/09/2007- 27/12/2019',fontsize=15,fontweight='bold',color='k')
plt.ylabel('1,000 VND',fontsize=17)
plt.xlabel('Source: VNDirect Securities Com.',fontsize=17,fontweight='normal',color='k')


# # ---------------------------------------The End-------------------------------------
