#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split



################################################### PREPARE ################################################### 

def get_cosmetic_data():
    '''This function creates dataframes from the csv files of the data.'''
    df1 = pd.read_csv('cosmetics.csv')

    df2 = pd.read_csv('cscpopendata.csv')

    return df1, df2

def prep_df1(df1):


    # change the df1 columns to lowercase
    df1.columns = df1.columns.str.lower()

    # turn the multiple ingredients in single cell into a list
    df1.ingredients = df1.ingredients.str.split(',')

    #explode those lists
    df1 = df1.explode('ingredients')

    #change the column values to lowercase
    df1['ingredients'] = df1['ingredients'].str.lower()

    #remove the whitespace from the column values
    df1['ingredients'] = df1['ingredients'].str.strip()