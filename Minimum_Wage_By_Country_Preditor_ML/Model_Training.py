import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import explained_variance_score as evs


# Data Import 
df = pd.read_csv('world-data-2023.csv')


# EDA
print(df.isnull().sum())
df.dropna(inplace= True)

print(df.dtypes)
la = LabelEncoder()

for i in df.columns:
    if df[i].dtype == 'object':
        df[i] = la.fit_transform(df[i])
        df[i] = df[i].astype('float')


sns.heatmap(df.corr(), vmin=0.0, vmax=1.0)
plt.show()

features = df[['Population', 'GDP', 'Total tax rate', 'Tax revenue (%)','Physicians per thousand', 'Gasoline Price', 'Currency-Code']]
target = df['Minimum wage']


# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size= 0.2, random_state= 42)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


# Model Training
models = [DecisionTreeRegressor(),
          RandomForestRegressor(),
          AdaBoostRegressor(),
          GradientBoostingRegressor(),
          SVR(),
          RandomForestRegressor()]

for m in models:
    print(m)
    m.fit(X_train,Y_train)

    pred_train = m.predict(X_train)
    print(f'Train Accuracy is : {evs(Y_train, pred_train)}')

    pred_test = m.predict(X_test)
    print(f'Test Accuracy is : {evs(Y_test, pred_test)}')