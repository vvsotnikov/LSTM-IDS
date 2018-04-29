import pandas as pd
from sklearn.preprocessing import LabelEncoder


def parse_dataset():
    # Default values.
    train_set = 'UNSW_NB15_training-set.csv'
    test_set = 'UNSW_NB15_testing-set.csv'
    training = pd.read_csv(train_set, index_col='id')
    testing = pd.read_csv(test_set, index_col='id')
    # To encode string  labels into numbers
    le = LabelEncoder()

    # Creates new dummy columns from each unique string in a particular feature
    training = pd.get_dummies(data=training, columns=['proto', 'service', 'state'])
    testing = pd.get_dummies(data=testing, columns=['proto', 'service', 'state'])

    # Making sure that the training features are same as testing features.
    # The training dataset has more unique protocols and states, therefore number \
    # of dummy columns will be different in both. We make it the same.
    traincols = list(training.columns.values)
    testcols = list(testing.columns.values)

    # For those in training but not in testing
    for col in traincols:
        # If a column is missing in the testing dataset, we add it
        if col not in testcols:
            testing[col] = 0
            testcols.append(col)
    # For those in testing but not in training
    for col in testcols:
        if col not in traincols:
            training[col] = 0
            traincols.append(col)

    # Moving the labels and categories to the end and making sure features are in the same order
    traincols.pop(traincols.index('attack_cat'))
    traincols.pop(traincols.index('label'))
    training = training[traincols + ['attack_cat', 'label']]
    testing = testing[traincols + ['attack_cat', 'label']]

    # Encoding the category names into numbers so that they can be one hot encoded later.
    training['attack_cat'] = le.fit_transform(training['attack_cat'])
    testing['attack_cat'] = le.fit_transform(testing['attack_cat'])

    # Normalising all numerical features:
    cols_to_normalise = list(training.columns.values)[:39]
    training[cols_to_normalise] = training[cols_to_normalise].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    testing[cols_to_normalise] = testing[cols_to_normalise].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    return training, testing, le
