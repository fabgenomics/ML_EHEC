#!/usr/bin/env python
# coding: utf-8

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("rtab")
parser.add_argument("csv")
args = parser.parse_args()

import pandas as pd

df = pd.read_csv(args.rtab, sep = "\t", header=0, usecols=['Gene', "x"])
df = df.set_index("Gene").transpose()
df = df.reset_index().rename(columns={'index': 'Gene'})
col_keep = ['Gene', 'Annotation']
df2 = pd.read_csv(args.csv, header=0, low_memory=False, usecols=col_keep)
locus_tag = df2["Annotation"]
locus_tag = locus_tag.str.split(pat = 'locus_tag=', n=1 , expand=True)
locus_tag = locus_tag[1]
locus_tag = locus_tag.str.split(pat=']', n=1 , expand=True).fillna('Empty')
locus_tag = locus_tag[0].tolist()
key_list2 = list(df2['Gene'])
dict_lookup2 = dict(zip(df2['Gene'], locus_tag))
red2 = df.columns.values
red2[1:] = [dict_lookup2[item] for item in key_list2]
df.columns = red2
df = df.groupby(by=df.columns, axis=1).sum()
df = df.drop(["Empty"], axis = 1)
list_features = pd.read_csv("list_of_features.txt", header=None).T
list_features.columns = list_features.iloc[0]

df2 = pd.concat([list_features, df], join = "inner")
df2 = df2.reset_index()
df2 = df2.drop([0])
df2 = pd.concat([df2, list_features], join = "outer").fillna(0)
df2 = df2.drop([0])
df2 = df2.rename(columns={"index" : "Gene"})
df2 = df2.astype(int)
df2[df2 > 1] = 1


import lightgbm as lgb
import xgboost as xgb
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score


clf_final = lgb.Booster(model_file = "Model_lgbm.txt")

with open(f'Logistic_Regression.pkl', 'rb') as f:
    clf_final2 = pickle.load(f)
with open(f'Decision_Tree.pkl', 'rb') as g:
    clf_final3 = pickle.load(g)
clf_final4 = xgb.XGBClassifier()
clf_final4.load_model("model_xgb.json")
with open(f'Random_Forest.pkl', 'rb') as i:
    clf_final5 = pickle.load(i)
with open(f'SVM.pkl', 'rb') as j:
    clf_final6 = pickle.load(j)
with open(f'Gradient_Boosting.pkl', 'rb') as k:
    clf_final7 = pickle.load(k)
with open(f'Extra_Trees.pkl', 'rb') as l:
    clf_final8 = pickle.load(l)
  
    
test = df2.drop(['Gene'], axis=1)
test = test.fillna(0)
test[test > 1] = 1
test = test.astype(int)
test_preds_dict = {}

# make predictions for each model and add the results to the dictionary
test_preds_dict['LGBM'] = clf_final.predict(test).flatten()
test_preds_dict['LR'] = clf_final2.predict_proba(test)[:,1].flatten()  # use the probability of positive class only
test_preds_dict['DT'] = clf_final3.predict_proba(test)[:,1].flatten()  # use the probability of positive class only
cols_when_model_builds = clf_final4.get_booster().feature_names
test = test[cols_when_model_builds]
test_preds_dict['XGBM'] = clf_final4.predict_proba(test)[:,1].flatten()  # use the probability of positive class only
test_preds_dict['RF'] = clf_final5.predict_proba(test)[:,1].flatten()  # use the probability of positive class only
test_preds_dict['SVM'] = clf_final6.predict_proba(test)[:,1].flatten()  # use the probability of positive class only
test_preds_dict['Gradient_Boosting'] = clf_final7.predict_proba(test)[:,1].flatten()  # use the probability of positive class only
test_preds_dict['ET'] = clf_final8.predict_proba(test)[:,1].flatten()  # use the probability of positive class only

# create a dataframe from the dictionary
df = pd.DataFrame(test_preds_dict)

# print the dataframe
print(df)

