import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

oecd_bli = pd.read_csv("oecd_bli_2015.csv",thousands=',')
gdp_per_capita = pd.read_csv("gdp_per_capita.csv",thousands=',',delimiter='\t',
encoding='latin1', na_values="n/a")
def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]
    country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
    X = np.c_[country_stats['GDP per capita']]
y = np.c_[country_stats ['Life satisfaction']]
country_stats.plot(kind='scatter',x = 'GDP per capita',y='Life satisfaction')
from sklearn.linear_moldel import LinearRegression
classifier = LinearRegression()
classifier.fit(X_train,y_train)
print('Accuracy of linear regression classifier on train set: {:.2f}'.format(classifier.score(X_train, y_train)))
print('Accuracy of linear regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))

#now let's try out the knn classifier 
from sklearn import preprocessing
from sklearn import utils
lab_enc = preprocessing.LabelEncoder()
training_scores_encoded = lab_enc.fit_transform(y_train)
print(training_scores_encoded)
print(utils.multiclass.type_of_target(y_train))
print(utils.multiclass.type_of_target(y_train.astype('int')))
print(utils.multiclass.type_of_target(training_scores_encoded))
from sklearn.neighbors import KNeighborsRegressor
#y_train = y_train
#y_test = y_test
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train,y_train)
print('accuracy of knn classifier on train set: %.2f ' % knn.score(X_train,y_train))
print('accuracy of knn classifier on test set: %.2f ' % knn.score(X_test,y_test))
