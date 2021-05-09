data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')
targets=data_train['Loan_Status'].map({'Y':1,'N':0})

combined = get_combined_data(data_train,data_test)
combined=impute_combined_data(combined)


combined=preprocess_property_data(combined)
combined=preprocess_combined_data(combined)
