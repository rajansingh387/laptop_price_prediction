
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('laptop_price_data.csv')

df.head()

df.isnull().sum()

#checking columns
df.columns

#converting all column names into lower case to avoid confusion
df.columns= df.columns.str.lower()
df.columns

df.info()
#investigating database checking null values and data types of columns

for i in df.columns:
    if df[i].nunique() < 10:
        print(f'The column "{i}" is __{df[i].dtype}__ \nhas __{df[i].nunique()}__ unique values: \n{df[i].value_counts()}')
        print(10*'==')
    else:
        print(f'The column "{i}" is __{df[i].dtype}__ \nhas __{df[i].nunique()}__ unique values')
        print(10*'==')

for i in df.columns:
    print(f'the columns "{i}" \nhas "{df[i].nunique()}" unique values ')
    print(30*'_')
#checking how many unique values each columns have using a for loop

df.shape
#checking rows and columns

df1= df.copy()

#because unnamed: 0 has 1302 unique values will not be usefull in our ml model so we are droping it for now
df.drop('unnamed: 0', axis= 1, inplace = True)

df.nunique()
#again checking unique values

#checking duplicates
df.duplicated().sum()

df.drop_duplicates(inplace= True)
# there are 30 duplicate values we are droping them

df.duplicated().sum()

"""**EDA** ***`Exploratory Data Analysis`***

*count of top 7 companies*
"""

a= df['company'].value_counts().sort_values(ascending= False)[:7]
a

a.plot(kind= 'barh')
plt.grid()

df.columns

df_c= df[['company', 'typename', 'ram', 'touchscreen', 'ips',
       'cpu brand', 'hdd', 'ssd', 'gpu brand']]

x= 0
fig= plt.figure(figsize=(20,15))
plt.subplots_adjust(wspace = 0.5)

for i in df_c.columns:
    ax = plt.subplot(331+x)
    ax = sns.countplot(data=df_c, y=i, color = '#A194B6')
    plt.grid(axis='x')
    ax.set_title(f'Distribution of {i}')
    x+=1

"""## **types of laptop and processor used**"""

plt.figure(figsize=(20,5))
sns.countplot(data= df, x= 'typename', hue= 'cpu brand')

plt.figure(figsize=(20,8))
sns.countplot(data= df, x= 'typename', hue= 'ram')
plt.xticks(rotation= 70)
plt.show()

plt.figure(figsize=(20,8))
sns.countplot(data= df, x= 'os', hue= 'company')
plt.xticks(rotation= 70)
plt.show()

plt.figure(figsize=(20,8))
sns.countplot(data= df, x= 'company', hue= 'gpu brand')
plt.xticks(rotation= 70)
plt.show()

df.head()

sns.boxplot(y=df['cpu brand'],x=df['price'])
plt.show()

sns.boxplot(y=df['gpu brand'],x=df['price'])
plt.show()

sns.boxplot(y=df['company'],x=df['price'])
plt.show()

sns.displot(df['price'])
plt.show()

corr = df.corr()

sns.heatmap(corr,annot=True,cmap='RdBu')
plt.show()

sns.heatmap(corr[abs(corr)>0.5],annot=True,cmap='RdBu')
plt.show()

df.describe(percentiles=[0.01,0.02,0.03,0.05,0.97,0.98,0.99]).T

print(df[df['ram']>32].shape)

print(df[df['weight']>4.42].shape)

x = df.drop('price',axis=1)
y = df['price']
print(type(x))
print(type(y))
print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

def eval_model(ytest,ypred):
    mae = mean_absolute_error(ytest,ypred)
    mse = mean_squared_error(ytest,ypred)
    rmse = np.sqrt(mse)
    r2s = r2_score(ytest,ypred)
    print('MAE',mae)
    print('MSE',mse)
    print('RMSE',rmse)
    print('R2 Score',r2s)

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

x_train.dtypes

step1= ColumnTransformer(transformers=[('ohe',OneHotEncoder(drop='first',sparse= False),[0,1,6,9,10])],remainder='passthrough')

step2= LinearRegression()
pipe_lr= Pipeline([('step1',step1),('step2',step2)])

pipe_lr.fit(x_train,y_train)
ypred_lr= pipe_lr.predict(x_test)
eval_model(y_test,ypred_lr)

step1 = ColumnTransformer(transformers=
                          [('ohe',OneHotEncoder(drop='first',sparse=False),[0,1,6,9,10])],
                              remainder='passthrough')
step2 = Ridge(alpha=2.1)

pipe_lr = Pipeline([('step1',step1),('step2',step2)])

pipe_lr.fit(x_train,y_train)

ypred_lr = pipe_lr.predict(x_test)

eval_model(y_test,ypred_lr)

step1 = ColumnTransformer(transformers=
                          [('ohe',OneHotEncoder(drop='first',sparse=False),[0,1,6,9,10])],
                              remainder='passthrough')
step2 = Lasso(alpha=2.1)

pipe_lr = Pipeline([('step1',step1),('step2',step2)])

pipe_lr.fit(x_train,y_train)

ypred_lr = pipe_lr.predict(x_test)

eval_model(y_test,ypred_lr)

step1 = ColumnTransformer(transformers=
                          [('ohe',OneHotEncoder(drop='first',sparse=False),[0,1,6,9,10])],
                              remainder='passthrough')
step2 = RandomForestRegressor(n_estimators=100,max_depth=15,min_samples_split=11,random_state=5)

pipe_rf = Pipeline([('step1',step1),('step2',step2)])

pipe_rf.fit(x_train,y_train)

ypred_rf = pipe_rf.predict(x_test)

eval_model(y_test,ypred_rf)

step1 = ColumnTransformer(transformers=
                          [('ohe',OneHotEncoder(drop='first',sparse=False),[0,1,6,9,10])],
                              remainder='passthrough')
step2 = DecisionTreeRegressor(max_depth=8,min_samples_split=11,random_state=5)

pipe_dt = Pipeline([('step1',step1),('step2',step2)])

pipe_dt.fit(x_train,y_train)

ypred_dt = pipe_dt.predict(x_test)

eval_model(y_test,ypred_dt)

import pickle

pickle.dump(pipe_rf,open('pipe_rf.pkl','wb'))
pickle.dump(df,open('data.pkl','wb'))
