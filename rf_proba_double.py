import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle


def confusion_calc(y1, y2, predictions1, predictions2):

	TP = sum((y2 == predictions2) & (y2 == True))*1.0

	TN1 = sum((y1 == predictions1) & (y1 == True))*1.0
	TN2 = sum((y2 == predictions2) & (y2 == False))*1.0
	TN = TN1 + TN2

	FP = sum((y2 != predictions2) & (predictions2 == True))*1.0

	FN1 = sum((y1 != predictions1) & (predictions1 == False))*1.0
	FN2 = sum((y2 != predictions2) & (predictions2 == False))*1.0
	FN = FN1 + FN2

	return TP, TN, FP, FN


def validation_metrics(TP, TN, FP, FN):

	print('TP = ', TP, ' TN = ', TN, ' FP = ', FP, ' FN = ', FN)

	sensitivity = TP / (TP + FN)
	precision = TP / (TP + FP)
	accuracy = (TP + TN) / (TP + TN + FP + FN)

	print('sensitivity: {}, precision: {}, accuracy: {}'.format(sensitivity, precision, accuracy))


def read_in_data(filepath, X_train, X_test, y_train, y_test):

	X_train = pd.read_csv(filepath + X_train)
	X_test = pd.read_csv(filepath + X_test)
	y_train = pd.read_csv(filepath + y_train)
	y_test = pd.read_csv(filepath + y_test)

	return X_train, X_test, y_train, y_test


def train_models(X_train, X_test, y_train, y_test):

	model = RandomForestClassifier(n_estimators = 500, n_jobs = -1)
	model.fit(X_train, y_train['target1'])

	pred_prob_mod1_train = model.predict_proba(X_train)
	pred_prob_mod1_test = model.predict_proba(X_test)

	train_mask = pred_prob_mod1_train[:,0] < 0.9
	test_mask = pred_prob_mod1_test[:,0] < 0.9

	y_train_2 = y_train['target2'][train_mask]
	y_test_2 = y_test['target2'][test_mask]
	X_train_2 = X_train[train_mask]
	X_test_2 = X_test[test_mask]

	second_model = RandomForestClassifier(n_estimators = 500, n_jobs = -1, oob_score = True)
	second_model.fit(X_train_2, y_train_2)

	return model, second_model, X_test_2, y_test_2, pred_prob_mod1_test, test_mask

if __name__ == '__main__':
	filepath = "data/"
	X_train, X_test, y_train, y_test = read_in_data(filepath, 'Xtrain.csv',
	 												'Xtest.csv', 'ytrain.csv', 'ytest.csv')

	results = train_models(X_train, X_test, y_train, y_test)
	model, second_model, X_test_2, y_test_2, pred_prob_mod1_test, pred_mod1_test = results

	pred_prob_mod2 = second_model.predict_proba(X_test_2)

	pred_mod2 = pred_prob_mod2[:,0] < 0.3

	print len(y_test['target1']), len(y_test_2), len(pred_mod1_test), len(pred_mod2)
	TP, TN, FP, FN = confusion_calc(y_test['target1'], y_test_2, pred_mod1_test, pred_mod2)
	validation_metrics(TP, TN, FP, FN)

	with open('model1.pkl', 'wb') as f:
		pickle.dump(model, f)
	with open('model2.pkl', 'wb') as f:
		pickle.dump(second_model, f)
