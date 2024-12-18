# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 17:01:03 2024

@author: haide
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

df = pd.read_csv('C:/Users/haide/Documents/Project2/tsa_claims.csv')

df['Item'].value_counts(normalize = True).cumsum()[:15].plot.barh()

df.info()
df.describe()



df.isna().mean().sort_values().plot.barh()
plt.title('Fraction NA by variable')
plt.xlabel('Fraction of values that are NA')
plt.ylabel('Variable')

print (df.dtypes)
print(df['Claim Amount'])


def clean_data(col):
    cleaned_data = df[col].str.replace("$", "", regex=False).str.replace(";","").str.replace(" ", "").replace("-",float("NaN"))
    return cleaned_data.convert_dtypes(float)

df["Claim Amount"] = clean_data('Claim Amount')
df['Close Amount'] = clean_data('Close Amount')
print(df["Claim Amount"])

df = df.dropna(subset = ['Close Amount'])
df = df.dropna(subset = ['Claim Amount'])
df = df.dropna(subset = ['Item'])

df['Item'].value_counts()
item_counts = df['Item'].value_counts()
items_to_keep = item_counts.head(10).index #change to top 10
df = df[df['Item'].isin(items_to_keep)]


plt.figure(figsize=(10, 6))
df['Item'].value_counts().plot.bar(rot = 45)
plt.xticks(ha='right', fontsize=10)
plt.tight_layout()  
plt.title('Items Lost or Damaged by TSA')

df['Claim Amount'] = pd.to_numeric(df['Claim Amount'], errors='coerce')
df['Close Amount'] = pd.to_numeric(df['Close Amount'], errors='coerce')

#df['Item'] = df['Item'].replace('Luggage (all types including footlockers)', 'Luggage')
#df['Item'] = df['Item'].replace('Clothing - Shoes; belts; accessories; etc.', 'Clothing')
#df['Item'] = df['Item'].replace('Jewelry - Fine', 'Jewelry')

df['Item'] = df['Item'].replace({
    'Luggage (all types including footlockers)': 'Luggage',
    'Clothing - Shoes; belts; accessories; etc.': 'Clothing',
    'Jewelry - Fine': 'Jewelry',    
    'Cameras - Digital': 'Cameras',
    'Computer - Laptop': 'Computer',
    'Eyeglasses - (including contact lenses)': 'Eyeglasses',
    'Cosmetics - Perfume; toilet articles; medicines; soaps; etc.': 'Cosmetics'
})


df['Claim Site'].value_counts()

top_2_claim_site = df['Claim Site'].value_counts().head(2).index
df = df[df['Claim Site'].isin(top_2_claim_site)]
top_2_claim_amount = df['Claim Amount'].value_counts().head(2).index
df = df[df['Claim Amount'].isin(top_2_claim_amount)]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
df['Claim Site'].value_counts().plot.barh(ax=axes[0], title='Claim Site')
df['Claim Type'].value_counts().plot.barh(ax=axes[1], title='Claim Type')
plt.tight_layout();    
plt.show()

df['Status'].value_counts().plot.barh()

df['Disposition'].value_counts().plot.barh()

df_S = df[df['Status'] == 'Denied']
(df_S['Disposition'].isna()).mean() 

denied_proportions = df.groupby('Disposition_NA')['Status'].apply(lambda x: (x == 'Denied').mean())
print(f"Proportion of 'Denied' cases:\n{denied_proportions}")

disposition_status_ct = pd.crosstab(df['Disposition_NA'], df['Status'], normalize='index')
print(disposition_status_ct)
disposition_status_ct.plot(kind='bar', stacked=True, figsize=(10, 6))

df['Disposition_NA'] = np.where(df['Disposition'].isna(), 1, 0)

status_na_proportion = df.groupby('Status')['Disposition_NA'].mean().sort_values(ascending=False)

print("Proportion of Disposition NA for each Status category:")
print(status_na_proportion)
status_na_proportion.plot.barh()


contingency_table = pd.crosstab(df['Disposition'].isna(), df['Close Amount'].isna())
print(contingency_table)

df['Disposition_NA'] = np.where(df['Disposition'].isna(), 1, 0)

# Group by Disposition_NA and analyze Status
status_correlation = df.groupby('Disposition_NA')['Status'].value_counts(normalize=True)
print(status_correlation)



print (df.dtypes)
print(df['Item'].value_counts())
df['Claim Amount'].max()
    
df = df[df['Claim Amount'] != 0]

top_05_percent_cutoff = df['Claim Amount'].quantile(0.995)
df = df[df['Claim Amount'] <= top_05_percent_cutoff]

plt.scatter(df['Claim Amount'], df['Close Amount'], alpha=0.5)
df[['Claim Amount', 'Close Amount']].boxplot()


df['CloseToClaimRatio'] =  (df['Close Amount'] / df['Claim Amount'])

df['CloseToClaimRatio'].mean()
df['CloseToClaimRatio'].max()
df['CloseToClaimRatio'].min()

df['CloseToClaimRatio'].plot(kind='kde', title='Age Density')

df['Claim Type'].value_counts().plot.barh()


df = pd.get_dummies(df, columns=['Item'], drop_first=True)

predictors = list(df.filter(like='Item_').columns)
target = 'CloseToClaimRatio'
X = df[predictors].values
y = df[target].values
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.30, random_state=42)

regr = LinearRegression()
regr.fit(X_train, y_train)

y_pred = regr.predict(X_test)
test_rmse = np.sqrt(((y_pred - y_test)**2).mean())
print('test RMSE: {:.3g}'.format(test_rmse))

mean_target = y_train.mean()
mse_baseline = ((mean_target - y_train)**2).mean()
rmse_baseline = np.sqrt(mse_baseline)
print(f"Baseline RMSE: {rmse_baseline:.1f}".format())

cv_scores = cross_val_score(regr, X, y, scoring='neg_mean_absolute_error', cv=10)
mean_cv_mae = -cv_scores.mean() 
print(f"Mean Cross-Validation MAE: {mean_cv_mae:.3g}")





