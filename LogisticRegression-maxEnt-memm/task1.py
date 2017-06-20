	#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nltk import NaiveBayesClassifier
from nltk import MaxentClassifier
from nltk.corpus import names
from nltk import classify


def gender_features(name):
	features = {}
	features["fl"] = name[0].lower()
	features["ll"] = name[-1].lower()
	features["fw"] = name[:2].lower()
	features["lw"] = name[-2:].lower()
	return features


our_names = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])

featuresets = [(gender_features(n), g) for (n, g) in our_names]

me_classifier = MaxentClassifier.train(featuresets,max_iter=40)

new_names = ([(name, 'male') for name in names.words('kazakh_male.txt')] + [(name, 'female') for name in names.words('kazakh_female.txt')])
test_set = [(gender_features(n), g) for (n, g) in new_names]

print format(classify.accuracy(me_classifier, test_set),'.5f')
