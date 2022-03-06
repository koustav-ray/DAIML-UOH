import os
import pickle 
import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# To load the data file and pickled model
_cwd = os.getcwd()
# generate path of data agnostic to OS
data_file = os.path.join(_cwd,'data','train.csv')
# generate path of model agnostic to OS
model_file = os.path.join(_cwd,'model','classifier.pkl')

# If the model and vectorizer are not already present, download them
# If the data file is not present
if not os.path.isfile(data_file): 
    # URL of the data file
    url = r'https://github.com/koustav-ray/DAIML-UOH/blob/main/data/train.csv?raw=true'
    # Download the data file
    resp = requests.get(url)
    # Open the data file to write the data
    with open(data_file, 'wb') as fopen:
        # Write the data	
        fopen.write(resp.content)

# If the model is not present
if not os.path.isfile(model_file):
    # URL of the model file
    url = r'https://github.com/koustav-ray/DAIML-UOH/blob/main/model/classifier.pkl?raw=true'
    # Download the model file
    resp = requests.get(url)
    # Open the model file to write the data
    with open(model_file, 'wb') as fopen:
        # Write the model
        fopen.write(resp.content)

# Load the data
with open(data_file, 'rb') as file:
    train_df = pd.read_csv(file, sep=r'\s*,\s*', na_values = -1)
# Load the pickled model
with open(model_file, 'rb') as file:
    classifier_model = pickle.load(file)

# Preprocessing of the data

# Replace comma(,) in "Field10" by blank("")
train_df["Field10"] = train_df["Field10"].apply(lambda x: x.str.replace(',',''))
train_df.replace('-1', "")

# Feature Encoding: Binary Mapping on Train data
for column in ['Field12', 'PersonalField7', 'PropertyField3', 'PropertyField4', 'PropertyField5', 'PropertyField30', 'PropertyField32', 'PropertyField34', 'PropertyField36', 'PropertyField37', 'PropertyField38', 'GeographicField63']:
    train_df[column] = train_df[column].apply(lambda x: 1 if x =='Y' else (0 if x =='N' else None)) 

# Feature Encoding: Label Encoding on Train data
for column in ['Field6', 'CoverageField8', 'CoverageField9', 'SalesField7', 'PropertyField7', 'PropertyField14', 'PropertyField28', 'PropertyField31', 'PropertyField33', 'PersonalField16', 'PersonalField17', 'PersonalField18', 'PersonalField19']:
    le = LabelEncoder()
    train_df[column] = le.fit_transform(train_df[column])

# One-hot Encoding: Label Encoding on Train data
# creating initial dataframe
GeographicField64_df = train_df['GeographicField64']
# generate binary values using get_dummies
dum_df = pd.get_dummies(GeographicField64_df, prefix='GeographicField64_is')
# drop 'GeographicField64'
train_df.drop(['GeographicField64'], axis = 1, inplace = True)
# merge with main df bridge_df on key values
train_df = train_df.join(dum_df)

# Change Original_Quote_Date to Date, Month, Year, day of Week on Train data
train_df['Original_Quote_Date'] = pd.to_datetime(train_df['Original_Quote_Date'])
train_df['Original_Quote_day'] = train_df['Original_Quote_Date'].dt.weekday
train_df['Original_Quote_date'] = train_df['Original_Quote_Date'].dt.day
train_df['Original_Quote_month'] = train_df['Original_Quote_Date'].dt.month
train_df['Original_Quote_year'] = train_df['Original_Quote_Date'].dt.year

# Drop the Redundant Columns on Train data
train_df.drop(['GeographicField10A','GeographicField21A','PropertyField11A','GeographicField60A','GeographicField5A','GeographicField61A','GeographicField22A','GeographicField56A','PropertyField2A','GeographicField62A','GeographicField18A','GeographicField23A','GeographicField14A','PropertyField29','CoverageField11B','CoverageField1B','CoverageField2A','CoverageField2B','CoverageField3A','CoverageField3B','CoverageField4A','CoverageField4B','GeographicField11A','GeographicField11B','GeographicField12A','GeographicField12B','GeographicField13A','GeographicField13B','GeographicField14B','GeographicField15A','GeographicField15B','GeographicField16A','GeographicField16B','GeographicField19B','GeographicField1B','GeographicField20A','GeographicField23B','GeographicField24B','GeographicField25B','GeographicField26B','GeographicField27A','GeographicField27B','GeographicField28B','GeographicField29B','GeographicField2B','GeographicField31B','GeographicField32B','GeographicField33B','GeographicField34B','GeographicField35B','GeographicField36B','GeographicField38B','GeographicField39B','GeographicField3B','GeographicField40B','GeographicField41B','GeographicField42B','GeographicField43B','GeographicField44B','GeographicField45B','GeographicField46B','GeographicField48A','GeographicField48B','GeographicField49B','GeographicField4B','GeographicField50B','GeographicField51B','GeographicField52A','GeographicField52B','GeographicField53B','GeographicField54B','GeographicField55B','GeographicField57B','GeographicField58B','GeographicField59B','GeographicField7B','GeographicField8A','GeographicField8B','GeographicField9A','GeographicField9B','PersonalField2','PersonalField23','PersonalField24','PersonalField25','PersonalField26','PersonalField32','PersonalField33','PersonalField35','PersonalField36','PersonalField37','PersonalField38','PersonalField41','PersonalField42','PersonalField43','PersonalField44','PersonalField45','PersonalField46','PersonalField47','PersonalField48','PersonalField4B','PersonalField50','PersonalField51','PersonalField52','PersonalField53','PersonalField55','PersonalField56','PersonalField57','PersonalField58','PersonalField60','PersonalField61','PersonalField62','PersonalField63','PersonalField68','PersonalField71','PersonalField72','PersonalField73','PersonalField75','PersonalField76','PersonalField77','PersonalField78','PersonalField80','PersonalField81','PersonalField82','PersonalField83','PropertyField16B','PropertyField1B','PropertyField21A','PropertyField21B','PropertyField24B','PropertyField26B','PropertyField39B','SalesField12','SalesField15','SalesField9','SalesField10','SalesField13','SalesField14','PersonalField11','PersonalField30','PersonalField39','PersonalField40','PersonalField49','PersonalField54','PersonalField59','PersonalField64','PersonalField65','PersonalField66','PersonalField67','PersonalField69','PersonalField70','PersonalField74','PersonalField79','PropertyField6','PropertyField9','PropertyField20','Original_Quote_Date'], axis = 1, inplace = True)

# Drop the target on Train data
train_df.drop(['QuoteConversion_Flag'], axis = 1)

# Impute Missing values using Simple Imputer on Train Data
X = train_df.copy(deep=True)
X.drop(['QuoteConversion_Flag'], axis = 1, inplace = True)
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X.iloc[:,:] = mean_imputer.fit_transform(X)

# Standardize the Train data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create a dictionary of QuoteNumber and Index Value
search_table = pd.Series(X.index,index=X.QuoteNumber.values).to_dict()

def predict_proba(input_id):
    """
    This function predicts the probability of the input quote being a successfully converted to purchase or not
    input: string
    Output: string
    """
    try:
        fetch_rowid = search_table.get(int(input_id))
        data_from_id = X_scaled[fetch_rowid,:].reshape(1,149)
        proba = classifier_model.predict_proba(data_from_id)
        return str(round(proba[0,1]*100,2))+' %'
    except:
        return -1
