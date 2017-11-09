import json
import pandas as pd
from sklearn.model_selection import train_test_split


def select_cols(df, listcols):
    df2 = df.copy()
    df2 = df2[listcols]
    return df2


def dummify_countries(df):
    df2 = df.copy()

    df2['country_US'] = df2['country'].isin(['US'])
    df2['country_English'] = df2['country'].isin(
        ['GB', 'CA', 'AU', 'NZ', 'IE'])
    df2['country_other'] = ~df2['country'].isin(
        ['US', 'GB', 'CA', 'AU', 'NZ', 'IE'])

    return df2


def add_targets(df):
    df2 = df.copy()
    df2['target1'] = df2['acct_type'].isin(['premium'])
    df2['target2'] = df2['acct_type'].isin(
        ['fraudster_event', 'fraudster', 'fraudster_att'])
    return df2


if __name__ == '__main__':
    df = pd.read_json('data/data.json')
    # df.to_csv('data/data.csv')

    df = dummify_countries(df)
    df = add_targets(df)

    features = ['channels', 'country_US', 'country_English', 'country_other', 'body_length', 'fb_published',
                'has_logo', 'name_length', 'num_order', 'sale_duration2', 'show_map', 'user_age', 'user_type']

    targets = ['target1', 'target2']
    X = select_cols(df, features)
    y = select_cols(df, targets)

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)

    Xtrain.to_csv('data/Xtrain.csv')
    ytrain.to_csv('data/ytrain.csv')
    Xtest.to_csv('data/Xtest.csv')
    ytest.to_csv('data/ytest.csv')
