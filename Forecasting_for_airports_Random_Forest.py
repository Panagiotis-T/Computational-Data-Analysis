import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import io

df = pd.read_excel(r'PATH', engine='openpyxl')

def make_plot(y, predi):
    plt.figure(figsize=(10,10))
    plt.scatter(y,predi, c='crimson')

    p1 = max(max(predi), max(y))
    p2 = min(min(predi), min(y))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')
    plt.show()


# Data Preparation

df = df.dropna(how='any')

df['day_week'] = df['ScheduleTime'].dt.dayofweek
df['day_month'] = df['ScheduleTime'].dt.day
df['month'] = df['ScheduleTime'].dt.month
df['week'] = df['ScheduleTime'].apply(lambda x: int(x.isocalendar()[1]))
df['year'] = df['ScheduleTime'].dt.year.apply(lambda x: 0 if x == 2021 else 1)

categorical_cols = ['AircraftType', 'FlightType', 'Sector', 'Airline', 'Destination']

# One-Hot Encoding
data = pd.get_dummies(df, columns = categorical_cols)
data["Flight_no_enc"] = data.groupby("FlightNumber")["LoadFactor"].transform("mean")

data = data.drop(columns=['ScheduleTime', 'FlightNumber','day_week','day_month','month'])

datay = data['LoadFactor']
dataX = data.drop(columns=['LoadFactor'])
X = dataX.to_numpy()
y = datay.to_numpy()

# Split train - test data

X_train = X[0:35840,:]
y_train = y[0:35840]

X_test = X[35840:,:]
y_test = y[35840:]

# define accuracy function

def accuracy_function(y_true, y_pred):
    idx = np.where(y_true != 0)[0]
    dev = 1 - y_pred[idx]/y_true[idx]
    acc = np.mean(1- np.abs(dev))
    return acc

# Build train Model

from sklearn.metrics import make_scorer
score = make_scorer(accuracy_function, greater_is_better=True)


import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Normalizer


n_estimators = [5,20,50,100,130,160] # number of trees in the random forest
max_features = ['auto', 'sqrt'] # number of features in consideration at every split
max_depth = [int(x) for x in np.linspace(20, 170, num = 10)] # maximum number of levels allowed in each decision tree
min_samples_split = [6, 10, 14] # minimum sample number to split a node
min_samples_leaf = [1, 3, 4] # minimum sample number that can be stored in a leaf node
bootstrap = [True, False] # method used to sample data points

random_grid = {'rgr__n_estimators': n_estimators,

'rgr__max_features': max_features,

'rgr__max_depth': max_depth,

'rgr__min_samples_split': min_samples_split,

'rgr__min_samples_leaf': min_samples_leaf,

'rgr__bootstrap': bootstrap}


pipe = Pipeline([
    ("rgr", RandomForestRegressor())
])

rf_random = RandomizedSearchCV(estimator = pipe ,param_distributions = random_grid,
               n_iter = 20, scoring = 'neg_mean_squared_error', cv = 5, verbose=10, random_state=35, n_jobs = -1)

rf_model = rf_random.fit(X_train, y_train)
print(rf_model)

print(rf_model.best_estimator_)

from sklearn.ensemble import BaggingRegressor


rf = RandomForestRegressor(max_depth=83, max_features='auto',
                                       min_samples_leaf=4, min_samples_split=14,
                                       n_estimators=160)
pipe = Pipeline([
    ("rgr", rf)
])

regr = BaggingRegressor(base_estimator=pipe,
                        n_estimators=10, random_state=0).fit(X_train, y_train)

# Make Predictions on the Test Set

predi = regr.predict(X_test)

accuracy_function(y_test,predi)

make_plot(y_test, predi)
