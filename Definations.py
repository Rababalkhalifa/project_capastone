"""
Created on Sun Apr 14 19:05:12 2013

@author1: Mohamed Aly <mohamed@mohamedaly.info>
@author2: Mahmoud Nabil <mah.nabil@yahoo.com>

Edited by @RababAlkhalifa <raalkhalifa@iau.edu.sa>

"""



import cPickle as pickle
import numpy as np
from AraTweet import *
import os

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn import metrics
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.ensemble.forest import RandomForestClassifier
from numpy.lib.scimath import sqrt
from numpy.ma.core import floor
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcess
from sklearn import svm
from sklearn import preprocessing
from pickle import FALSE
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from sklearn.manifold import Isomap
from sklearn.manifold import SpectralEmbedding
from sklearn.decomposition import TruncatedSVD

LoadValidation = True  # Load The validation set
Evaluate_On_TestSet = True  # Evaluate either on evaluation or on test set if LoadValidation is True
Extract_Features = False  # Apply Feature Extraction techniques
Two_Stages_Classification = False
CrossValidation = False
UseLexicon = False
# data sets
datas = [
    dict(name="4-balanced", params=dict(klass="4", balanced="balanced")),
    dict(name="4-unbalanced", params=dict(klass="4", balanced="unbalanced")),
]

import qalsadi.analex as analex
# tokenizer
an = analex.Analex()
tokenizer = an.text_tokenize

# features
Features_Generators = [
    dict(name="count_ng1",
         feat_generator=CountVectorizer(tokenizer=tokenizer, ngram_range=(1, 1))),
    dict(name="count_ng2",
         feat_generator=CountVectorizer(tokenizer=tokenizer, ngram_range=(1, 2))),
    dict(name="count_ng3",
         feat_generator=CountVectorizer(tokenizer=tokenizer, ngram_range=(1, 3))),
    dict(name="tfidf_ng1",
         feat_generator=TfidfVectorizer(tokenizer=tokenizer, ngram_range=(1, 1))),
    dict(name="tfidf_ng2",
         feat_generator=TfidfVectorizer(tokenizer=tokenizer, ngram_range=(1, 2))),
    dict(name="tfidf_ng3",
         feat_generator=TfidfVectorizer(tokenizer=tokenizer, ngram_range=(1, 3))),
]

# classifiers
classifiers = [  
    
    dict(name="Logistic Regression", parameter_tunning=True, tune_clf=GridSearchCV(LogisticRegression(), [{ 'C': [1.00000000e+00, 2.78255940e+00, 7.74263683e+00, 2.15443469e+01,
       5.99484250e+01, 1.66810054e+02, 4.64158883e+02, 1.29154967e+03,
       3.59381366e+03, 1.00000000e+04]}],  cv=3), clf=LogisticRegression( dual=False , random_state=0)),
    dict(name="Passive Aggresive", parameter_tunning=True, clf=PassiveAggressiveClassifier(random_state=0) 
        ,tune_clf=GridSearchCV(PassiveAggressiveClassifier() , {'C': [1.0 , .7 , .5 , .3 , .1], 
                                'loss': ['hinge'],
                                'n_iter': [15],
                                'n_jobs': [1],
                                'random_state': [0],
                                'shuffle': [True,False],
                                'verbose': [0 ],
                                'warm_start': [False , True]} )),
    dict(name="SVM", parameter_tunning=True, clf=LinearSVC(tol=1e-3, random_state=0) ,tune_clf=GridSearchCV(LinearSVC() , {'C': [1.0 , .7 , .5 , .3 , .1], 
                                'dual': [True],
                                'fit_intercept': [True , False],
                                'loss': ['hinge','squared_hinge'],
                                'max_iter': [1000],
                                'random_state': [0],
                                'multi_class': ['ovr', 'crammer_singer'],
                                'intercept_scaling': [.3 , .5 ,1 ],
                                'fit_intercept': [True,False]} )),
    dict(name="Perceptron", parameter_tunning=True, clf=Perceptron(random_state=0) , tune_clf=GridSearchCV( Perceptron(), {'alpha': [1, .8 , .5 , .3, .1 , .01 , .001 , .0001 ], 
                                        'fit_intercept': [True,False],
                                        'n_iter': [5, 10 , 15],
                                        'random_state': [0],
                                        'penalty': ['l1','l2'],
                                        'shuffle': [True , False],
                                        'warm_start': [True,False],
                                        'eta0': [1.0 , .5 ]
                                } )),
    #
    dict(name="bnb", parameter_tunning=True, clf=BernoulliNB(),tune_clf=GridSearchCV(BernoulliNB() , {'alpha': [1.0 , .8 , .6 , .4 , .2], 
                                'binarize': [0.0 , .25 , .5], 
                                'fit_prior': [True,False] } )),
    dict(name="sgd", parameter_tunning=False, clf=SGDClassifier(loss="hinge", penalty="l2", random_state=0)),
    dict(name="KNN", parameter_tunning=False, tune_clf=GridSearchCV(KNeighborsClassifier(),
        [{'n_neighbors': [5, 10, 50, 100], 'metric': ['euclidean', 'minkowski'], 'p': [2, 3, 4, 5]}], cv=5),
         clf=KNeighborsClassifier(n_neighbors=5, metric='euclidean', random_state=0)),
    dict(name='RandomForest', parameter_tunning=False, clf=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=110, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=2, min_samples_split=20,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=0, verbose=0, warm_start=False)),
               
    dict(name='DecisionTree', parameter_tunning=False, clf=DecisionTreeClassifier()),
    
        dict(name="mnb", parameter_tunning=True, tune_clf=GridSearchCV(MultinomialNB(),{'alpha': [1.0, .1 , .3 , .6 , .8 , 0.0], 'fit_prior': [True,False]}, cv=5) , clf=MultinomialNB() ),

]







