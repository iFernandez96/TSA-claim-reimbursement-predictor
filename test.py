# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 11:42:04 2024

@author: haide
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.model_selection import learning_curve, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import tree
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
import graphviz
from sklearn.tree import export_graphviz

df = pd.read_csv('tsa_claims.csv')
#df = pd.read_csv('tsa_claims.csv')
df['Item'].value_counts(normalize = True).cumsum()[:15].plot.barh()


df.info()
df.describe()

print (df.dtypes)
print(df['Claim Amount'])


def clean_data(col):
    cleaned_data = df[col].str.replace("$", "", regex=False).str.replace(";","").str.replace(" ", "").replace("-",float("NaN"))
    return cleaned_data.convert_dtypes(float)

df["Claim Amount"] = clean_data('Claim Amount')
df['Close Amount'] = clean_data('Close Amount')
print(df["Claim Amount"])
val1 = df.isna().sum()

df = df.dropna(subset = ['Close Amount'])
df = df.dropna(subset = ['Claim Amount'])
val2 = df.isna().sum()
df = df.drop('Airline Name', axis=1)

df = df.dropna(subset = ['Item'])
df['Item'].value_counts()
item_counts = df['Item'].value_counts()
items_to_keep = item_counts.head(10).index #change to top 10
df = df[df['Item'].isin(items_to_keep)]
df = df[df['Item'] != 'Other']

plt.figure(figsize=(10, 6))
df['Item'].value_counts().plot.bar(rot = 45)
plt.xticks(ha='right', fontsize=10)
plt.tight_layout()  
plt.title('Items Lost or Damaged by TSA')

# df = pd.get_dummies(df, columns=['Item'], drop_first=True)

df['Claim Amount'] = pd.to_numeric(df['Claim Amount'], errors='coerce')
df['Close Amount'] = pd.to_numeric(df['Close Amount'], errors='coerce')
#stats.zscore(df.groupby("Item")[['Claim Amount', 'Close Amount']].mean()).sort_values(by='Close Amount', ascending = False).plot.barh()


#lower_bound = df['Claim Amount'].mean() - 3 * df['Claim Amount'].std()
#upper_bound = df['Claim Amount'].mean() + 3 * df['Claim Amount'].std()

#df = df[(df['Claim Amount'] >= lower_bound) & (df['Claim Amount'] <= upper_bound)]

df = df[df['Claim Amount'] != 0]
df['CloseToClaimRatio'] =  (df['Close Amount'] / df['Claim Amount'])

df['CloseToClaimRatio'].mean()
df['CloseToClaimRatio'].max()
df['CloseToClaimRatio'].min()

print(df.info())

df = pd.get_dummies(df, columns=['Item'], drop_first=True)

predictors = ['Claim Amount'] + list(df.filter(like='Item_').columns)
target = 'CloseToClaimRatio'
X = df[predictors].values
y = df[target].values
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.30, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

regr = LinearRegression()
clf = DecisionTreeRegressor(max_depth=2, random_state=0)
clf.fit(X_train, y_train)
# regr.fit(X_train, y_train)
dot_data = export_graphviz(clf, precision=2, feature_names=predictors, proportion=True, class_names=[target], filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph

y_pred = regr.predict(X_test)
test_rmse = np.sqrt(((y_pred - y_test)**2).mean())
print('test RMSE: {:.3g}'.format(test_rmse))

mean_target = y_train.mean()
mse_baseline = ((mean_target - y_test)**2).mean()
rmse_baseline = np.sqrt(mse_baseline)
print(f"Baseline RMSE: {rmse_baseline:.1f}".format())

sns.scatterplot(y_test, y_pred)
plt.plot(color='grey', linestyle='dashed')
plt.xlabel('actual')
plt.ylabel('predicted')

scores = cross_val_score(regr, X, y, cv=5, scoring='neg_root_mean_squared_error')
print('Cross-Validated RMSE:', -scores.mean())

MAE = abs(y_pred - y_test).mean()
print('Mean Absolute Error:', MAE)

