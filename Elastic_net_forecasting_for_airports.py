import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def make_plot(y1,predi):
    plt.figure(figsize=(10,10))
    plt.scatter(y1,predi, c='crimson')

    p1 = max(max(predi), max(y1))
    p2 = min(min(predi), min(y1))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')
    plt.show()

df1 = pd.read_excel(r'D:\Documents\DTU/02582 Computational Data Analysis/Case 1/Realized Schedule 20210101-20220228.xlsx')
df2 = pd.read_excel(r'D:\Documents\DTU/02582 Computational Data Analysis/Case 1/Future Schedule 20220301-20220331.xlsx')
df = pd.concat([df1, df2], ignore_index=True)
df = df.sort_values(by=['ScheduleTime'])

df['day_week'] = df['ScheduleTime'].dt.dayofweek
df['day_month'] = df['ScheduleTime'].dt.day
df['month'] = df['ScheduleTime'].dt.month
df['week'] = df['ScheduleTime'].apply(lambda x: int(x.isocalendar()[1]))
df['year'] = df['ScheduleTime'].dt.year.apply(lambda x: 0 if x == 2021 else 1)

def hour_transformer(t, type_):
    ts = t.hour + t.minute / 60
    x = 2 * np.pi * ts/23.99
    if type_ == 'sin':
        cycl_time = np.sin(x)
    elif type_ == 'cos':
        cycl_time = np.cos(x)
    return cycl_time

def general_transformer(t, max_scale, type_):
    x = 2 * np.pi * t / max_scale
    if type_ == 'sin':
        cycl_time = np.sin(x)
    elif type_ == 'cos':
        cycl_time = np.cos(x)
    return cycl_time


df['time_sin'] = df['ScheduleTime'].apply(hour_transformer,args=('sin',))
df['time_cos'] = df['ScheduleTime'].apply(hour_transformer,args=('cos',))

df['day_week_sin'] = df['day_week'].apply(general_transformer,args=(6,'sin',))
df['day_week_cos'] = df['day_week'].apply(general_transformer,args=(6,'cos',))

df['day_month_sin'] = df['day_month'].apply(general_transformer,args=(31,'sin',))
df['day_month_cos'] = df['day_month'].apply(general_transformer,args=(31,'cos',))

df['month_sin'] = df['month'].apply(general_transformer,args=(12,'sin',))
df['month_cos'] = df['month'].apply(general_transformer,args=(12,'cos',))

categorical_cols = ['AircraftType', 'FlightType', 'Sector', 'Airline', 'Destination']
data = pd.get_dummies(df, columns = categorical_cols)
data["Flight_no_enc"] = data.groupby("FlightNumber")["LoadFactor"].transform("mean")
data["Flight_no_enc"] = data["Flight_no_enc"].replace(np.nan, 0.5)


data = data[data['ScheduleTime'].dt.date>datetime.date(2021,5,31)]
data = data.drop(columns=['ScheduleTime', 'FlightNumber','day_week','day_month','month'])

print(data)

data_for = data[data['LoadFactor'].isna()]
data_fore = data_for.drop('LoadFactor', 1)
data = data.dropna(how='any')

datay = data['LoadFactor']
dataX = data.drop(columns=['LoadFactor'])

X = dataX.to_numpy()
y = datay.to_numpy()

X_train = X[:31208,:]
y_train = y[:31208]

X_test = X[31208:,:]
y_test = y[31208:]

from sklearn import linear_model 
from sklearn.model_selection import KFold
import warnings

def accuracy_function(y_true, y_pred):
    idx = np.where(y_true != 0)[0]
    dev = 1 - y_pred[idx]/y_true[idx]
    acc = np.mean(1- np.abs(dev))
    return acc


from sklearn.metrics import make_scorer
import warnings
score = make_scorer(accuracy_function, greater_is_better=True)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit

model = linear_model.ElasticNet(normalize=True)


parametersGrid = {"alpha": np.logspace(-7, -3, num=5),
                      "l1_ratio": np.arange(0.6, 0.9, 0.1)}

grid = GridSearchCV(model, parametersGrid, scoring='neg_mean_squared_error', cv=5, verbose=10)
with warnings.catch_warnings(): # done to disable all the convergence warnings from elastic net
    warnings.simplefilter("ignore")
    grid.fit(X_train, y_train)

print(grid.best_estimator_)

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingRegressor

with warnings.catch_warnings(): # done to disable all the convergence warnings from elastic net
    warnings.simplefilter("ignore")

    model2 = linear_model.ElasticNet(l1_ratio = 0.6, alpha=1e-05, normalize=True)
    regr = BaggingRegressor(base_estimator=model2,
                        n_estimators=10, random_state=0).fit(X_train, y_train)
    

predi = regr.predict(X_test)

accuracy_function(y_test,predi)

make_plot(y_test,predi)