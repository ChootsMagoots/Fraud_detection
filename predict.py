import pickle
from feature_eng import select_cols
from feature_eng import dummify_countries
from feature_eng import add_num_previous_payouts
from feature_eng import add_targets

#This is the command to read in the model:
simplemodel = pickle.load( open( "simplemodel.pkl", "rb" ) )

#This reads in the data:
df_t = pd.read_json('data/data.json')
#And picks one row:
df = df_t.iloc[[0]]

#And now, we clean the data:
df = dummify_countries(df)
df = add_targets(df)
df = add_num_previous_payouts(df)

features = ['channels', 'country_US', 'country_English', 'country_other', 'body_length', 'fb_published',
            'has_logo', 'name_length', 'num_order', 'sale_duration2', 'show_map', 'user_age', 'user_type', 'num_prev_payouts']

X = select_cols(df, features)
y = select_cols(df, 'target2')

#Here's the prediction for that row:
predictions = simplemodel.predict(X)
