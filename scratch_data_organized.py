# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 20:00:22 2024

@author: israe
"""

# All import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree, DecisionTreeRegressor
from sklearn.model_selection import learning_curve, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import tree
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, mean_squared_error

df = pd.read_csv('tsa_claims.csv');

df.info()

df.isna().mean().sort_values().plot.barh()
plt.title('Fraction NA by variable')
plt.xlabel('Fraction of values that are NA')
plt.ylabel('Variable');

df['Disposition_NA'] = np.where(df['Disposition'].isna(), 1, 0)
status_na_proportion = df.groupby('Status')['Disposition_NA'].mean().sort_values(ascending=False)
print("Proportion of Disposition NA for each Status category:")
print(status_na_proportion)

contingency_table = pd.crosstab(df['Disposition'].isna(), df['Close Amount'].isna())
print(contingency_table)

df['Item'].value_counts()


def clean_data(col):
    cleaned_data = df[col].str.replace("$", "", regex=False).str.replace(";","").str.replace(" ", "").replace("-",np.nan)
    return cleaned_data.convert_dtypes(float)
df["Claim Amount"] = clean_data('Claim Amount')
df['Close Amount'] = clean_data('Close Amount')

df = df.dropna(subset = ['Close Amount'])
df = df.dropna(subset = ['Claim Amount'])
df = df.dropna(subset = ['Disposition'])
df = df.dropna(subset = ['Item'])
df = df[df['Item'] != 'Other']

item_counts = df['Item'].value_counts()
items_to_keep = item_counts.head(15).index
df = df[df['Item'].isin(items_to_keep)]
df = df[df['Item'] != 'Other']

df['Claim Amount'] = pd.to_numeric(df['Claim Amount'], errors='coerce')
df['Close Amount'] = pd.to_numeric(df['Close Amount'], errors='coerce')

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
df['Claim Site'].value_counts().plot.barh(ax=axes[0], title='Claim Site')
df['Claim Type'].value_counts().plot.barh(ax=axes[1], title='Claim Type')
plt.tight_layout();

top_2_claim_site = df['Claim Site'].value_counts().head(2).index
df = df[df['Claim Site'].isin(top_2_claim_site)]
top_2_claim_amount = df['Claim Amount'].value_counts().head(2).index
df = df[df['Claim Amount'].isin(top_2_claim_amount)]

top_05_percent_cutoff = df['Claim Amount'].quantile(0.995)
df = df[df['Claim Amount'] <= top_05_percent_cutoff]

df['Item'] = df['Item'].replace({
    'Luggage (all types including footlockers)': 'Luggage',
    'Clothing - Shoes; belts; accessories; etc.': 'Clothing',
    'Jewelry - Fine': 'Jewelry',    
    'Cameras - Digital': 'Cameras',
    'Computer - Laptop': 'Computer',
    'Eyeglasses - (including contact lenses)': 'Eyeglasses',
    'Cosmetics - Perfume; toilet articles; medicines; soaps; etc.': 'Cosmetics',
    'Sporting Equipment & Supplies (footballs; parachutes; etc.)': 'Sporting Equipment',
    'MP3 Players-(iPods; etc)': 'MP3 Players',
    'DVD/CD Players': 'DVD/CD Players',
    'Cell Phones': 'Cell Phones',
    'Currency': 'Currency',
    'Medicines': 'Medicines',
    'Locks': 'Locks'
})

df['Item'].value_counts(normalize = True).cumsum().head(15).plot.barh()
plt.ylabel('Item')
plt.xlabel('Cumulative Proportion of Items')
plt.title('Cumulative Sum Distribution of Items');

df.groupby("Item")[['Claim Amount', 'Close Amount']].mean().sort_values(by="Claim Amount").head(15).plot.barh()
plt.xlabel('Dollar Amounts')
plt.title('Item Claim and Close amounts sorted by Claim Amounts');


df.groupby("Item")[['Claim Amount', 'Close Amount']].mean().sort_values(by="Close Amount").head(15).plot.barh()
plt.xlabel('Dollar Amounts')
plt.title('Item Claim and Close amounts sorted by Close Amounts');

grouped = df.groupby("Item")[["Claim Amount", "Close Amount"]].mean()
grouped["Percentage"] = (grouped["Close Amount"] / grouped["Claim Amount"])
grouped = grouped.sort_values(by="Percentage")
grouped["Percentage"].tail(15).plot.barh()
plt.xlabel('Percentage (%)')
plt.title('Percentage of Close Amount based on Claim Amount by Item');

df = pd.get_dummies(df, columns=['Item'], drop_first=True)

df['CloseToClaimRatio'] =  (df['Close Amount'] / df['Claim Amount'])

# Assign Predictor and Target variables
predictors = ['Claim Amount'] + list(df.filter(like='Item_').columns)
target = 'CloseToClaimRatio'

X = df[predictors].values
y = df[target].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2']
}


# Train decision tree regressor
regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X_train, y_train)

grid_search = GridSearchCV(
    estimator=regressor,
    param_grid=param_grid,
    cv=10,  # 5-fold cross-validation
    scoring='neg_mean_squared_error',  # Optimize for Mean Squared Error
)

# Perform the grid search
grid_search.fit(X_train, y_train)

# Evaluate the model
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Test Root Mean Squared Error: {rmse:.2f}")

baseline_pred = np.full_like(y_test, y_train.mean())  # Mean prediction baseline
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
print(f"Baseline RMSE: {baseline_rmse}")

# Plot decision tree
plt.figure(figsize=(20, 10))
plot_tree(regressor, feature_names=predictors, filled=True, fontsize=10)
plt.title("Decision Tree Regressor")

regr = LinearRegression()
regr.fit(X_train, y_train)

y_pred = regr.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('test RMSE: {:.3g}'.format(test_rmse))

mean_target = y_train.mean()
mse_baseline = ((mean_target - y_test)**2).mean()
rmse_baseline = np.sqrt(mse_baseline)
print(f"Baseline RMSE: {rmse_baseline:.1f}".format())