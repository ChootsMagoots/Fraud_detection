import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
	df = pd.read_json('data/data.json')
	small_df = df.drop(['acct_type', 'approx_payout_date', 'channels', 'description', 'event_created', 'event_end', 'event_published', 'event_start', 'gts', 'has_analytics', 'listed', 'name', 'num_payouts', 'object_id', 'org_desc', 'org_name', 'payee_name', 'venue_address', 'venue_latitude', 'venue_longitude', 'venue_name', 'country', 'currency', 'email_domain', 'payout_type', 'previous_payouts', 'ticket_types', 'venue_country', 'venue_state', 'delivery_method', 'has_header', 'org_facebook', 'org_twitter', 'sale_duration'], axis = 1)
	target = df['acct_type']
	target = pd.get_dummies(target)
	premiums = target.pop('premium')
	fraud = target[['fraudster', 'fraudster_att', 'fraudster_event']]
	fraud = fraud.sum(axis = 1)

	X_train, X_test, y_train, y_test = train_test_split(small_df, premiums, test_size = 0.20)

	model = RandomForestClassifier(n_estimators = 2000, n_jobs = -1)
	model.fit(small_df, premiums)

	predictions = model.predict(small_df)

	# y_test_array = np.array(premiums)
	# TP, TN, FP, FN = 0, 0, 0, 0
	# for thing1, thing2 in zip(predictions, y_test_array):
	# 	if thing2 == 1 and thing1 - thing2 == 0:
	# 		TP += 1
	# 	elif thing2 == 0 and thing1 - thing2 == 0:
	# 		TN += 1
	# 	elif thing2 == 0 and thing1 - thing2 == 1:
	# 		FP += 1
	# 	else: 
	# 		FN += 1

	# print('TP = ', TP, ' TN = ', TN, ' FP = ', FP, ' FN = ', FN)

	# sensitivity = TP / (TP + FN)
	# precision = TP / (TP + FP)
	# accuracy = (TP + TN) / (TP + TN + FP + FN)

	# print('sensitivity: {}, precision: {}, accuracy: {}'.format(sensitivity, precision, accuracy))

	mask = premiums == predictions
	second_df = small_df[predictions != mask]
	fraud = fraud[predictions != mask]

	X2_train, X2_test, fraud_train, fraud_test = train_test_split(second_df, fraud, test_size = 0.20)

	second_model = RandomForestClassifier(n_estimators = 2000, n_jobs = -1, oob_score = True)
	second_model.fit(X2_train, fraud_train)

	predictions2 = second_model.predict(X2_test)

	y_test_array = np.array(fraud_test)
	TP, TN, FP, FN = 0, 0, 0, 0
	for thing1, thing2 in zip(predictions2, y_test_array):
		if thing2 == 1 and thing1 - thing2 == 0:
			TP += 1
		elif thing2 == 0 and thing1 - thing2 == 0:
			TN += 1
		elif thing2 == 0 and thing1 - thing2 == 1:
			FP += 1
		else: 
			FN += 1

	print('TP = ', TP, ' TN = ', TN, ' FP = ', FP, ' FN = ', FN)

	sensitivity = TP / (TP + FN)
	precision = TP / (TP + FP)
	accuracy = (TP + TN) / (TP + TN + FP + FN)

	print('sensitivity: {}, precision: {}, accuracy: {}'.format(sensitivity, precision, accuracy))

	i = 0
	for j, entry in enumerate(predictions): 
		if entry == 1: 
			continue
		if predictions2[i] == 0:
			predictions[j] = 1
		i += 1


