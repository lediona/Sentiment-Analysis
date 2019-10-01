import pandas as pd
import numpy as np

"""
    A set of helper functions for the Take home assignment of RubiKloud
    """


def read_in_dataset(dset, verbose=False):
    
    """Read in one of the datasets (train or properties)
        
        Keyword arguments:
        dset -- a string
        verbose -- whether or not to print info about the dataset
        
        Returns:
        a pandas dataframe
        """
    
    df =  pd.read_csv('{0}.csv'.format(dset), encoding = "ISO-8859-1")
    
    if verbose:
        print('\n{0:*^80}'.format(' Reading in the {0} dataset '.format(dset)))
        print("\nit has {0} rows and {1} columns".format(*df.shape))
        print('\n{0:*^80}\n'.format(' It has the following columns '))
        print(df.columns)
        print('\n{0:*^80}\n'.format(' The first 5 rows look like this '))
        print(df.head())
    
    return df



def scores(y_test, y_pred):
    
    precision, recall, fscore, support = score(y_test, y_pred)
    
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))

def dist(lat1, long1, lat2, long2):
    """
        the function that calculates the euclidian distance of the lat and long of two points
        """
    return np.sqrt((lat1-lat2)**2+(long1-long2)**2)


def find_closest_city(lat, long):
    
    distances = cities_df.apply(lambda row: dist(lat, long, row['latitude'], row['longitude']), axis=1)
    
    return cities_df.loc[distances.idxmin(), 'name']



