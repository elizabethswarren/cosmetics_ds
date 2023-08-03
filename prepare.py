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
import sklearn.preprocessing
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

import category_encoders as ce



################################################### ACQUIRE ################################################### 

def get_cosmetic_data():
    '''This function creates dataframes from the csv files of the data.'''
    df1 = pd.read_csv('cosmetics.csv')

    df2 = pd.read_csv('cscpopendata.csv')

    return df1, df2

################################################### PREPARE ################################################### 

def prep_df1(df1, df2):

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

    #change the values in column to lowercase in second df
    df2['ChemicalName'] = df2['ChemicalName'].str.lower()

    #remove the whitespace in column values in second df
    df2['ChemicalName'] = df2['ChemicalName'].str.strip()

    #create a bool list if ingredients in df1 is in the hazardous chemicals lists
    bad_ingredients = df1['ingredients'].isin(df2['ChemicalName'])

    #pull out the entire rows
    has_ingredient = df1.loc[bad_ingredients]

    #create the target variable column for df1
    df1['has_hazard_ingredient'] = df1['ingredients'].isin(has_ingredient['ingredients'])

    #drop extra columns
    df1 = df1.drop(columns=['price', 'rank', 'combination', 'dry', 'normal', 'oily', 'sensitive'])

    #drop ingredient column
    df1 = df1.drop(columns='ingredients') 

    #sort by has_hazard_ingredients and then drop the 
    df1 = df1.sort_values(by='has_hazard_ingredient').drop_duplicates(keep='last')

    #make a list of the product names
    has_hazard_name = df1[df1.has_hazard_ingredient == True].name.to_list()

    #separate the products that are actually true hazard
    actual_true = df1[df1.name.isin(has_hazard_name) & df1.has_hazard_ingredient == True]

    #separate the products that are actually false hazard
    actual_false = df1[(df1.has_hazard_ingredient == False) & ~(df1.name.isin(has_hazard_name))]

    #combine them
    df1 = pd.concat([actual_true, actual_false], ignore_index=True)

   # rename values to match
    df1.label = df1.label.replace({'Moisturizer': 'Skin Care Products',
                      'Cleanser': 'Skin Care Products',
                      'Face Mask': 'Skin Care Products',
                      'Treatment': 'Skin Care Products',
                      'Eye cream': 'Skin Care Products',
                      'Sun protect': 'Sun-Related Products'})
    
    #rename columns
    df1.rename(columns={'label':'type'}, inplace=True)
    
    return df1, df2

def prep_df2(df2):

    #lowercase column names
    df2.columns = df2.columns.str.lower()

    #drop duplicates
    df2 = df2.drop_duplicates()

    #drop duplicate cphid
    df2 = df2.drop_duplicates(subset=['cdphid'])

    #add target variable
    df2['has_hazard_ingredient'] = True

    #drop columns
    df2 = df2.drop(columns=['cdphid', 'csfid', 'csf', 'companyid', 'companyname', 'primarycategoryid', 'subcategoryid', 'subcategory', 'casid', 'casnumber', 'chemicalid', 'chemicalname', 'initialdatereported', 'mostrecentdatereported', 'discontinueddate', 'chemicalcreatedat', 'chemicaldateremoved', 'chemicalupdatedat', 'chemicalcount'])

    #rename columns
    df2.rename(columns={'productname':'name', 'brandname': 'brand', 'primarycategory': 'type'}, inplace = True)

    return df2

def final_prep(df1, df2):

    #concat the two df
    df = pd.concat([df2, df1])

    #drop duplicates
    df = df.drop_duplicates()

    #remove non alphanumeric values in name column
    df.name = df.name.str.replace('[^a-zA-Z0-9\s]', '', regex=True)

    # remove non alphanumeric values in brand column
    df.brand = df.brand.str.replace('[^a-zA-Z0-9\s]', '', regex=True)

    #reset index
    df = df.reset_index(drop=True)

    #change target to int type
    df['has_hazard_ingredient'] = df['has_hazard_ingredient'].astype(int)

    #clean up type column
    df.type = df.type.str.strip()

    #isolate skincare products
    skincare = df.loc[df['type']=='Skin Care Products']

    #isolate suncare products
    suncare = df.loc[df['type']=='Sun-Related Products']

    #final df
    df = pd.concat([skincare, suncare], ignore_index=True).reset_index(drop=True)

    return df

def train_validate_test_split(df, target, seed=611):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, 
                                            stratify=df[target])
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       stratify=train_validate[target])
    return train, validate, test

def encode_df(train, validate, test):

    enc_leave = ce.leave_one_out.LeaveOneOutEncoder(cols=['brand','type'],return_df=True,sigma=0.5)

    train_encode = enc_leave.fit_transform(X=train,y=train['has_hazard_ingredient'])

    validate_encode = enc_leave.transform(X=validate,y=validate['has_hazard_ingredient'])

    test_encode = enc_leave.transform(X=test)

    X_train = train_encode.drop(columns=['has_hazard_ingredient', 'name'])
    y_train = train_encode.has_hazard_ingredient

    X_val = validate_encode.drop(columns=['has_hazard_ingredient','name'])
    y_val = validate_encode.has_hazard_ingredient

    X_test = test_encode.drop(columns=['has_hazard_ingredient','name'])
    y_test = test_encode.has_hazard_ingredient

    return X_train, y_train, X_val, y_val, X_test, y_test


def model_results(X_train, y_train, X_val, y_val):

    # create model
    clf = DecisionTreeClassifier(max_depth=3, random_state=611)

    # fit the model`
    clf = clf.fit(X_train, y_train)

    # accuracy on train data
    print('Accuracy of Decision Tree classifier on training set: {:.2f}'
      .format(clf.score(X_train, y_train)))

    #accuracy on validate data
    print('Accuracy of Decision Tree classifier on validate set: {:.2f}'
      .format(clf.score(X_val, y_val)))

    # create the model
    rf = RandomForestClassifier(max_depth=3, min_samples_leaf=3, random_state=611)

    # fit the model
    rf = rf.fit(X_train, y_train)

    # accuracy for train data
    print('\n\nAccuracy of random forest classifier on training set: {:.2f}'
     .format(rf.score(X_train, y_train)))

    # accuracy for validate data
    print('Accuracy of random forest classifier on validate set: {:.2f}'
     .format(rf.score(X_val, y_val)))
    
    # create model
    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')

    # fit the model
    knn = knn.fit(X_train, y_train)

    # accuracy for training data
    print('\n\nAccuracy of KNN classifier on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))

    #accuracy for validate data
    print('Accuracy of KNN classifier on validate set: {:.2f}'
     .format(knn.score(X_val, y_val)))
    
    # create model
    logit = LogisticRegression(C=1, class_weight={0: 1, 1: 99}, random_state=611)

    #  fit model
    logit = logit.fit(X_train, y_train)

    # accuracy of train data
    print('\n\nAccuracy of Logistic Regression classifier on training set: {:.2f}'
     .format(logit.score(X_train, y_train)))

    # accuracy of validate data
    print('Accuracy of Logistic Regression classifier on validate set: {:.2f}'
     .format(logit.score(X_val, y_val)))


def final_model(X_train, y_train, X_test, y_test):

     # create model
    clf = DecisionTreeClassifier(max_depth=3, random_state=611)

    # fit the model`
    clf = clf.fit(X_train, y_train)

    # accuracy of rf on test data
    print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))