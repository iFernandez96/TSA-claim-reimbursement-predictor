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
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


df = pd.read_csv('C:/Users/haide/Documents/Project2/tsa_claims.csv')
df['Item'].value_counts(normalize = True).cumsum()[:15].plot.bar()


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
stats.zscore(df.groupby("Item")[['Claim Amount', 'Close Amount']].mean()).sort_values(by='Close Amount', ascending = False).plot.barh()


lower_bound = df['Claim Amount'].mean() - 3 * df['Claim Amount'].std()
upper_bound = df['Claim Amount'].mean() + 3 * df['Claim Amount'].std()

df = df[(df['Claim Amount'] >= lower_bound) & (df['Claim Amount'] <= upper_bound)]

lower_bound = df['Close Amount'].mean() - 3 * df['Close Amount'].std()
upper_bound = df['Close Amount'].mean() + 3 * df['Close Amount'].std()

df = df[(df['Close Amount'] >= lower_bound) & (df['Close Amount'] <= upper_bound)]



print(df.info())

df = pd.get_dummies(df, columns=['Item'], drop_first=True)

#df['Close Amount'] = np.log1p(df['Close Amount'])

predictors = ['Claim Amount'] + list(df.filter(like='Item_').columns)
target = 'Close Amount'
X = df[predictors].values
y = df[target].values
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.30, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsRegressor()
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
test_rmse = np.sqrt(((y_pred - y_test)**2).mean())
print('test RMSE: {:.3g}'.format(test_rmse))

mean_target = y_train.mean()
mse_baseline = ((mean_target - y_test)**2).mean()
rmse_baseline = np.sqrt(mse_baseline)
print(f"Baseline RMSE: {rmse_baseline:.1f}".format())


plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.title('Predicted vs Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

plt.hist(y_pred, bins=30)
plt.title('Distribution of Predicted Values')
plt.show()

scores = cross_val_score(knn, X, y, cv=5, scoring='neg_root_mean_squared_error')
print('Cross-Validated RMSE:', -scores.mean())

