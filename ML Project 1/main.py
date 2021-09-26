import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load
from sklearn.metrics import mean_squared_error

def print_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation:", scores.std())



# reading of data from csv file
housing = pd.read_csv('data.csv')

# Normal Train test splitting of data
# train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# Stratified Train test splitting of data
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.iloc[train_index]
    strat_test_set = housing.iloc[test_index]

# Construction of a pipeline for our data
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    # ...... as many more
    ('std_scalar', StandardScaler()),
])


housing = strat_train_set.drop("MEDV", axis=1)
housing_labels = strat_train_set["MEDV"].copy()


housing_num_tr = my_pipeline.fit_transform(housing)
print(housing_num_tr.shape)

# Model selection
# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()

model.fit(housing_num_tr, housing_labels)
some_data = housing_num_tr[:5]
some_labels = housing_labels[:5]

prepared_data = my_pipeline.fit_transform(some_data)
# print(model.predict(prepared_data))
# print(list(some_labels))

scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)

print_scores(rmse_scores)


dump(model, "Dragon.joblib")


# Testing of the model
x_test = strat_test_set.drop("MEDV", axis=1)
y_test = strat_test_set["MEDV"].copy()
x_test_prepared = my_pipeline.transform(x_test)
final_prediction = model.predict(x_test_prepared)
final_mse = mean_squared_error(y_test, final_prediction)
final_rmse = np.sqrt(final_mse)
# print(final_prediction, list(y_test))


# Using the model
model = load('Dragon.joblib')
features = np.array([-10.43942006,  0.12628155, -1.12165014, -7.27288841, -19.42262747,
       -10.24065257, -1.31238772,  0.61111401, -5.0016859 , -10.5778192 ,
       -10.97491834,  0.41164221, -10.86091034])
print(model.predict([features]))