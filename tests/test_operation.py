import pandas as pd
from app import x_train_new

def test_data_processing():
    # Checking if the DataFrame has been correctly normalized
    assert x_train_new.shape[1] == 4  # Assuming the dataset has 4 features
    assert x_train_new.describe().loc['min'].min() >= 0  # Check minimum value is >= 0
    assert x_train_new.describe().loc['max'].max() <= 1  # Check maximum value is <= 1

    # Additional checks on column types, column names, etc.
    expected_columns = ['age', 'gender', 'fever', 'cough', 'city']
    assert list(x_train_new.columns) == expected_columns
