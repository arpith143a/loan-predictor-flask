import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import numpy as np
import pickle


data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')
targets=data_train['Loan_Status'].map({'Y':1,'N':0})

le = LabelEncoder()

def get_combined_data(train,test):

    targets = train.Loan_Status
    train.drop('Loan_Status', 1, inplace=True)
    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop(['index', 'Loan_ID'], inplace=True, axis=1)

    return combined

def impute_combined_data(combined):

    combined['Gender'].fillna('Male', inplace=True)
    combined['Married'].fillna('Yes', inplace=True)
    combined['Self_Employed'].fillna('No', inplace=True)
    combined['Dependents'].fillna(combined['Dependents'].mode()[0],inplace=True)
    combined['LoanAmount'].fillna(combined['LoanAmount'].median(),inplace=True)
    combined['Credit_History'].fillna(1,inplace=True)
    combined['Loan_Amount_Term'].fillna(combined['Loan_Amount_Term'].median(),inplace=True)

    return combined


def preprocess_property_data(combined):

    property_dummies = pd.get_dummies(combined['Property_Area'], prefix='Property')
    combined = pd.concat([combined, property_dummies], axis=1)
    combined.drop('Property_Area', axis=1, inplace=True)
    combined['Dependents'] = le.fit_transform(combined['Dependents'])

    return combined


def preprocess_combined_data(combined):

    combined['Gender'] = combined['Gender'].map({'Male':1,'Female':0})
    combined['Married'] = combined['Married'].map({'Yes':1,'No':0})
    combined['Education'] = combined['Education'].map({'Graduate':1,'Not Graduate':0})
    combined['Self_Employed'] = combined['Self_Employed'].map({'Yes':1,'No':0})
    combined['TotalIncome'] = combined['ApplicantIncome'] + combined['CoapplicantIncome']

    return combined

def preprocess_new(comb):

    if comb[-1]=="Urban":
            comb.pop()
            comb.append(1)
            comb.append(0)
            comb.append(0)
    elif comb[-1]=="Semiurban":
            comb.pop()
            comb.append(0)
            comb.append(1)
            comb.append(0)
    elif comb[-1]=="Rural":
            comb.pop()
            comb.append(0)
            comb.append(0)
            comb.append(1)
    comb[2]=comb[2].strip("+")
    print(comb[2])
    comb = [int(i) for i in comb]
    comb.append(comb[5]+comb[6])
    # print(len(comb))

    return comb

combined = get_combined_data(data_train,data_test)
combined=impute_combined_data(combined)

combined=preprocess_property_data(combined)
combined=preprocess_combined_data(combined)

transformation=MinMaxScaler().fit(combined[['Dependents','LoanAmount','TotalIncome','Loan_Amount_Term']])
combined[['Dependents','LoanAmount','TotalIncome','Loan_Amount_Term']]=transformation.transform(combined[['Dependents','LoanAmount','TotalIncome','Loan_Amount_Term']])

def compute_score(clf, X, y, scoring='accuracy'):

    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)

    return np.mean(xval)

def recover_train_test_target(combined):

    train = combined.head(614)
    del train['ApplicantIncome'],train['CoapplicantIncome']
    test = combined.iloc[614:]

    return train, test

train, test = recover_train_test_target(combined)

def feature_selection(train,test):

    clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
    clf = clf.fit(train, targets)
    features = pd.DataFrame()
    features['Feature'] = train.columns
    features['Importance'] = clf.feature_importances_
    features.sort_values(by=['Importance'], ascending=False, inplace=True)
    features.set_index('Feature', inplace=True)

    model = SelectFromModel(clf, prefit=True)
    train_reduced = model.transform(train)
    test_reduced = model.transform(test)

def build_model(train,targets):

    parameters = {'bootstrap': False,
                  'min_samples_leaf': 3,
                  'n_estimators': 50,
                  'min_samples_split': 10,
                  'max_features': 'sqrt',
                  'max_depth': 6}

    model = RandomForestClassifier(**parameters)
    model.fit(train, targets)

    return model


# model=build_model(train,targets)
# filename = 'finalized_model.sav'
# pickle.dump(model, open(filename, 'wb'))
