import json
import pandas as pd
from sklearn.model_selection import train_test_split

def select_cols(df, listcols):
    df2 = df.copy()
    df2 = df2[listcols]
    return df2

def dummify_countries(df):
    df2 = df.copy()
    df2['country_US'] = 0
    if(df2['country'] == 'US'):
        df2['country_US'] = 1
    df2['country_English'] = 0
    if(df2['country'] in ['GB', 'CA', 'AU', 'NZ', 'IE']):
        df2['country_English'] = 1
    df2['country_other'] = 0
    if((df2['country_US'] == 0) and (df2['country_English'] == 0)):
        df2['country_other'] = 1
    return df2


if __name__ == '__main__':
    df = pd.read_json('data/data.json')
    #df.to_csv('data/data.csv')

    features = ['channels', 'country_US', 'country_English', 'country_other', 'body_length', 'fb_published', 'has_logo', 'name_length', 'num_order', 'sale_duration2', 'show_map', 'user_age', 'user_type']

    df = dummify_countries(df)

    targets = ['acct_type']
    X = select_cols(df, features)
    y = select_cols(df, targets)

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)



    Xtrain.to_csv('Xtrain.csv')
    ytrain.to_csv('ytrain.csv')
    Xtest.to_csv('Xtest.csv')
    ytest.to_csv('ytest.csv')





    #