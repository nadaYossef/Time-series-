#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install statsmodels')
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

import matplotlib.pyplot as plt
from matplotlib import rcParams
from cycler import cycler

rcParams['figure.figsize'] = 18, 5
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['axes.prop_cycle'] = cycler(color=['#365977'])
rcParams['lines.linewidth'] = 2.5


# In[10]:


df = pd.read_csv("C:/Users/Lenovo/Desktop/LTOTALNSA.csv", index_col='DATE', parse_dates=True)
df.tail()


# In[11]:


df.describe()


# In[13]:


df.shape


# In[12]:


df.info()


# In[14]:


plt.figure(figsize=(14, 4))
plt.title('LW Vehicle Sales in Thousands of Units', size=20)
plt.plot(df)


# In[15]:


from statsmodels.tsa.seasonal import seasonal_decompose
plt.rcParams['figure.figsize'] = 12, 8
seasonal_decompose(df).plot();


# In[16]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing


# In[17]:


model = ExponentialSmoothing(
    endog=df['LTOTALNSA'],
    trend='add',
    seasonal='add',
    seasonal_periods=12
).fit()


# In[18]:


forecasts = model.forecast(steps=24)


# In[20]:


plt.figure(figsize=(14, 4))
plt.title('LW Vehicle Sales Forecasts', size=20, weight="bold")
plt.plot(df[-240:], label='History')
plt.plot(forecasts, label='Forecasts', color='red')
plt.legend()


# In[21]:


from datetime import datetime, timedelta
datetime(year=2022, month=12, day=3)


# In[22]:


datetime(year=2022, month=12, day=3, hour=13, minute=59, second=44)


# In[23]:


datetime.now()


# In[24]:


datetime.now() - timedelta(days=1)


# In[26]:


now = datetime.now()

print(now.year, now.month, now.day, now.hour, now.minute, now.second, sep=",")


# In[27]:


date_list = [
    datetime(2022, 12, 1),
    datetime(2022, 12, 2),
    datetime(2022, 12, 3)
]
date_list


# In[28]:


import numpy as np

date_list = np.array(['2022-12-01', '2022-12-02', '2022-12-03'], dtype='datetime64')
date_list


# In[29]:


date_list = np.array(['2022-12-01', '2022-12-02', '2022-12-03'], dtype='datetime64[s]')
date_list


# In[30]:


date_list = np.array(['2022-12-01', '2022-12-02', '2022-12-03'], dtype='datetime64[Y]')
date_list


# In[31]:


date_list = np.arange('2022-12-01', '2022-12-05', dtype='datetime64[D]')
date_list


# In[32]:


date_list = pd.date_range(start='2022-12-01', end='2022-12-05')
date_list


# In[33]:


date_list = pd.date_range(start='2022-12-01', periods=10, freq='D')
date_list


# In[34]:


print(date_list.min())
print(date_list.max())


# In[35]:


yearly_totals = df.resample(rule='Y').sum()
yearly_totals.head()


# In[36]:


quarterly_means = df.resample(rule='Q').mean()
quarterly_means


# In[37]:


df_shift = df.copy()

df_shift['Shift_1'] = df_shift['LTOTALNSA'].shift(1)
df_shift['Shift_2'] = df_shift['LTOTALNSA'].shift(2)

df_shift.head()



# In[38]:


df_shift = df.copy()

df_shift['Shift_Neg_1'] = df_shift['LTOTALNSA'].shift(-1)
df_shift['Shift_Neg_2'] = df_shift['LTOTALNSA'].shift(-2)

df_shift.tail()


# In[39]:


df_rolling = df.copy()
df_rolling['Quarterly_Rolling'] = df_rolling['LTOTALNSA'].rolling(window=3).mean()
df_rolling['Yearly_Rolling'] = df_rolling['LTOTALNSA'].rolling(window=12).mean()
df_rolling.head(15)


# In[40]:


df_diff = df.copy()

df_diff['Diff_1'] = df_diff['LTOTALNSA'].diff(periods=1)
df_diff['Diff_2'] = df_diff['LTOTALNSA'].diff(periods=2)

df_diff.head()


# In[41]:


df.plot(title='Light weight vehicle sales');


# In[42]:


plt.title('Light weight vehicle sales', size=20)
plt.xlabel('Time period', size=14)
plt.ylabel('Number of units sold in 000', size=14)

plt.plot(df['LTOTALNSA']);


# In[43]:


plt.title('Light weight vehicle sales', size=20)
plt.xlabel('Time period', size=14)
plt.ylabel('Number of units sold in 000', size=14)
plt.plot(df['LTOTALNSA']['1990-01-01':'2005-01-01']);


# In[44]:


plt.title('Light weight vehicle sales', size=20)
plt.xlabel('Time period', size=14)
plt.ylabel('Number of units sold in 000', size=14)
plt.xlim(np.array(['1990-01-01', '2005-01-01'], dtype='datetime64'))
plt.ylim([0, 2000])
plt.plot(df['LTOTALNSA']);



# In[45]:


white_noise = np.random.randn(1000)
plt.title('White noise plot', size=20)
plt.plot(np.arange(len(white_noise)), white_noise);


# In[48]:


white_noise_chunks = np.split(white_noise, 20)
means, stds = [], []

for chunk in white_noise_chunks:
    means.append(np.mean(chunk))
    stds.append(np.std(chunk))
    
plt.title('White noise mean and standard deviation comparison', size=20)
plt.plot(np.arange(len(means)), [white_noise.mean()] * len(means), label='Global mean', lw=1.5)
plt.scatter(x=np.arange(len(means)), y=means, label='Mean', s=100)
plt.plot(np.arange(len(stds)), [white_noise.std()] * len(stds), label='Global std', lw=1.5, color='red')
plt.scatter(x=np.arange(len(stds)), y=stds, label='STD', s=100, color='red')
plt.legend();


# In[51]:


from IPython import display
display.set_matplotlib_formats("svg")
random_walk = [0]

for i in range(1, 1000):
    num = -1 if np.random.random() < 0.5 else 1
    random_walk.append(random_walk[-1] + num)
    
plt.title('Random walk plot', size=20)
plt.plot(np.arange(len(random_walk)), random_walk);


# In[53]:


s_random_walk = pd.Series(random_walk)
s_random_walk_diff = s_random_walk.diff().dropna()

plt.title('Random walk first order difference', size=20)
plt.plot(np.arange(len(s_random_walk_diff)), s_random_walk_diff);


# In[ ]:




