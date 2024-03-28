# required pakages @@

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
import statsmodels.api as sm

data = pd.read_csv("cars_price.csv")


# data cleaning

data.info()
missing_count = data.isnull().sum()
print(missing_count)
data.replace('?', pd.NA, inplace=True)
df_cleaned = data.dropna()


df_cleaned.info()

df_cleaned['price'] = pd.to_numeric(df_cleaned['price'], errors='coerce')
target = df_cleaned['price']


# seperating categorical_columns & numerical_columns  from main dataset
categorical_columns = df_cleaned.loc[:, ['symboling','make','fuel-type','aspiration','num-of-doors','body-style','drive-wheels','engine-location','engine-type','num-of-cylinders','fuel-system']]

numerical_columns = df_cleaned.loc[:, ['normalized-losses','wheel-base','length','width','height','curb-weight','engine-size','bore','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg']]

# correlation_matrix numerical_columns and target column price
numerical_columns['price'] = target
new_df = numerical_columns.copy()
df = pd.DataFrame(new_df)
correlation_matrix = df.corr()
correlation_matrix = df.corr()

# Create a heatmap using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.show()

df_cleaned = df_cleaned.drop(columns=['price'])


# creating dummy columns for categorical_columns and concate with numerical_columns

df = pd.DataFrame(categorical_columns)

# Convert 'symboling' to categorical data type
df['symboling'] = df['symboling'].astype('category')

# Create dummy variables for categorical columns
categorical_columns = ['symboling', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 
                       'drive-wheels', 'engine-location', 'engine-type', 'num-of-cylinders', 'fuel-system']

dummy_columns = pd.get_dummies(df[categorical_columns], prefix=categorical_columns, drop_first=True)

# Concatenate the dummy columns with the original DataFrame
df = pd.concat([df_cleaned, dummy_columns], axis=1)

# Drop the original categorical columns
df.drop(columns=categorical_columns, inplace=True)

print(df)


# scaling the features
from sklearn.preprocessing import scale
scaled_features = scale(df)
X_scaled = pd.DataFrame(scaled_features, columns=df.columns)

print(X_scaled)


X = X_scaled
y = target

# LinearRegression on whole dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.7,test_size = 0.3, random_state=100)
# instantiate
lm = LinearRegression()
# fit
lm.fit(X_train, y_train)
# predict 
y_pred = lm.predict(X_test)
# metrics
from sklearn.metrics import r2_score
print(r2_score(y_true=y_test, y_pred=y_pred))


# calculating and visualize adjusted_r2 for feature selection

# ... (load and preprocess your data)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lists to store results
n_features_list = list(range(2, 58))  # Specify the range from 2 to 20
adjusted_r2 = []
r2 = []
test_r2 = []

# Iterate over different number of features
for n_features in n_features_list:
    # RFE with n features
    lm = LinearRegression()
    rfe_n = RFE(estimator=lm, n_features_to_select=n_features)
    rfe_n.fit(X_train, y_train)
    col_n = X_train.columns[rfe_n.support_]
    X_train_rfe_n = X_train[col_n]
    X_train_rfe_n = sm.add_constant(X_train_rfe_n)
    
    # Reset indices to ensure proper alignment
    X_train_rfe_n.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    
    lm_n = sm.OLS(y_train, X_train_rfe_n).fit()
    adjusted_r2.append(lm_n.rsquared_adj)
    r2.append(lm_n.rsquared)
    
    X_test_rfe_n = X_test[col_n]
    X_test_rfe_n = sm.add_constant(X_test_rfe_n, has_constant='add')
    y_pred = lm_n.predict(X_test_rfe_n)
    test_r2.append(r2_score(y_test, y_pred))

# Plotting
plt.figure(figsize=(10, 8))
plt.plot(n_features_list, adjusted_r2, label="adjusted_r2")
plt.plot(n_features_list, r2, label="train_r2")
plt.plot(n_features_list, test_r2, label="test_r2")
plt.legend(loc='upper left')
plt.xlabel('Number of Features')
plt.ylabel('R-squared')
plt.title('Number of Features vs R-squared')
plt.grid(True)
plt.show()


# feature selection and final LinearRegression model



# ... (load and preprocess your data)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# Number of features to select
n_features = 31

# RFE with n features
lm = LinearRegression()
rfe_n = RFE(estimator=lm, n_features_to_select=n_features)
rfe_n.fit(X_train, y_train)
col_n = X_train.columns[rfe_n.support_]
X_train_rfe_n = X_train[col_n]
X_train_rfe_n = sm.add_constant(X_train_rfe_n)

# Reset indices to ensure proper alignment
X_train_rfe_n.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)

lm_n = sm.OLS(y_train, X_train_rfe_n).fit()

X_test_rfe_n = X_test[col_n]
X_test_rfe_n = sm.add_constant(X_test_rfe_n, has_constant='add')
y_pred = lm_n.predict(X_test_rfe_n)

# Calculate R-squared score for the test set
r2_test = r2_score(y_test, y_pred)
print(f'R-squared score for the test set: {r2_test:.4f}')

# Print the summary of the trained model
print(lm_n.summary())


# showing r2_test , mse, mae

from sklearn.metrics import mean_squared_error, mean_absolute_error

r2_test = r2_score(y_test, y_pred)
print(f'R-squared score for the test set: {r2_test:.4f}')
# Calculate Mean Squared Error (MSE) for the test set
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error (MSE) on test set: {mse:.4f}')

# Calculate Mean Absolute Error (MAE) for the test set
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error (MAE) on test set: {mae:.4f}')




## brif description:

# r-square score is a statistical measure that represents the proportion of the variance in the dependent variable that is explained by the independent variables in a regression model.


# r-square score of 0.9161 means that approximately 91.61% of the variance in the dependent variable can be explained by the independent variables in this regression model.

# The MSE is a measure of the average squared difference between the predicted values and the actual values in your test dataset.

# An MSE of 3298805.7910 means that, on average, the squared difference between the predicted values and the actual values is approximately 3,298,805.7910.

# The MAE is a measure of the average absolute difference between the predicted values and the actual values in your test dataset.


# A MAE of 1344.6137 means that, on average, the absolute difference between the predicted values and the actual values is approximately 1344.6137.







    
    






