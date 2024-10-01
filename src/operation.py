import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("data/covid.csv")

# Perform initial processing
df = df.dropna()

# Label encoding
lb = LabelEncoder()
df['gender'] = lb.fit_transform(df['gender'])
df['cough'] = lb.fit_transform(df['cough'])
df['city'] = lb.fit_transform(df['city'])
df['has_covid'] = lb.fit_transform(df['has_covid'])

# Splitting the data
x = df.drop(columns=['has_covid'])
y = df['has_covid']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Normalizing the data
mn = MinMaxScaler()
x_train_mn = mn.fit_transform(x_train)

# Convert back to DataFrame for testing purposes
x_train_new = pd.DataFrame(x_train_mn, columns=x_train.columns)

print(np.round(x_train_new.describe(), 2))
