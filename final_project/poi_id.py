#!/usr/bin/python

import sys
import pickle
import numpy as np
import pandas as pd

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = list(data_dict['METTS MARK'].keys())
features_list.remove('poi')
features_list.remove('email_address')
features_list.remove('total_payments')
features_list.remove('total_stock_value')
features_list.remove('other')


### Task 2: Remove outliers
data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')


# Remove columns with > 50% NaN's
df = pd.DataFrame(data_dict).T
df.replace(to_replace='NaN', value=np.nan, inplace=True)
for key in features_list:
    if df[key].isnull().sum() > df.shape[0] * 0.5:
        features_list.remove(key)
features_list = ['poi'] + features_list




### Store to my_dataset for easy export below.
my_dataset = data_dict




### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scl_features = scaler.fit_transform(features)


# from sklearn.preprocessing import RobustScaler
# scaler = RobustScaler()
# scl_features = scaler.fit_transform(features)



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# adapted from: http://www.codiply.com/blog/hyperparameter-grid-search-across-multiple-models-in-scikit-learn/


from sklearn.model_selection import GridSearchCV




class EstimatorSelectionHelper:
    """
    http://www.davidsbatista.net/blog/2018/02/23/model_optimization/
    https://stackoverflow.com/questions/23045318/scikit-grid-search-over-multiple-classifiers

    """    
    def __init__(self, models, params):
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}
    
    def fit(self, X, y, **grid_kwargs):
        for key in self.keys:
            print('Running GridSearchCV for %s.' % key)
            model = self.models[key]
            params = self.params[key]
            grid_search = GridSearchCV(model, params, **grid_kwargs)
            grid_search.fit(X, y)
            self.grid_searches[key] = grid_search
        print('Done.')
    
    def score_summary(self, sort_by='mean_test_score'):
        frames = []
        for name, grid_search in self.grid_searches.items():
            frame = pd.DataFrame(grid_search.cv_results_)
            frame = frame.filter(regex='^(?!.*param_).*$')
            frame['estimator'] = len(frame)*[name]
            frames.append(frame)
        df = pd.concat(frames)
        
        df = df.sort_values([sort_by], ascending=False)
        df = df.reset_index()
        df = df.drop(['rank_test_score', 'index'], 1)
        
        columns = df.columns.tolist()
        columns.remove('estimator')
        columns = ['estimator']+columns
        df = df[columns]
        return df



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)



models1 = {
    'RandomForestClassifier': RandomForestClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'SVC': SVC(),
    'MLPClassifier': MLPClassifier()
}

params1 = {
    'RandomForestClassifier': { 'n_estimators': [8, 16, 32] },
    'AdaBoostClassifier':  { 'n_estimators': [8, 16, 32, 50], 'learning_rate': [0.1, 0.5, 1.0]},
    'GradientBoostingClassifier': { 'n_estimators': [16, 32, 50], 'learning_rate': [0.8, 1.0] },
    'MLPClassifier': {'hidden_layer_sizes': (32,16), 'learning_rate_init': [0.001]},
    'SVC': [
        {'kernel': ['rbf'], 'C': [1, 10, 100]},
    ]
}

helper1 = EstimatorSelectionHelper(models1, params1)
helper1.fit(features_train, labels_train, scoring='f1', n_jobs=24, cv=10)

summary = helper1.score_summary()

print(summary)
# print(summary.mean_test_score)

#### select the best
clf = AdaBoostClassifier(n_estimators=32, learning_rate=1.0)

clf.fit(features_train, labels_train)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)