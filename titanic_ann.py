import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder, StandardScaler


def transform_data(data):
    df = data.copy()
    df['Title'] = df.Name.str.split(',').str.get(1).str.split('.').str.get(0).str.strip()
    df.Embarked = df.Embarked.fillna('S')
    df.Age = df.Age.fillna(df.Age.dropna().mean())
    df.Fare = df.Fare.fillna(0)
    df = pd.get_dummies(df, columns=['Pclass', 'Embarked'])

    return df


def drop_unneeded_columns(data):
    df = data.copy()
    df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

    return df


def missing_data(data):
    print('Total null values per column')
    print(data.isnull().sum(), end='\n\n')
    print('Percentage of null values per column')
    print(round(data.isnull().sum() / len(data) * 100, 2), end='\n\n')


# load the data
df_train_loaded = pd.read_csv('data/train.csv')
df_test_loaded = pd.read_csv('data/test.csv')
# df_gender_submission = pd.read_csv('data/gender_submission.csv')

# Transform data
df_train = transform_data(df_train_loaded)
df_train = drop_unneeded_columns(df_train)

df_test = transform_data(df_test_loaded)
df_test = drop_unneeded_columns(df_test)

# Missing data
missing_data(df_train)
missing_data(df_test)

# Split DataFrame into X and y
X_train = df_train.drop('Survived', axis=1).values
y_train = df_train.Survived.values

X_test = df_test.values  # y_test is what we need to predict

# Encode categorical data
le = LabelEncoder()

X_train[:, 0] = le.fit_transform(X_train[:, 0])  # Sex
X_train[:, 5] = le.fit_transform(X_train[:, 5])  # Title

X_test[:, 0] = le.fit_transform(X_test[:, 0])  # Sex
X_test[:, 5] = le.fit_transform(X_test[:, 5])  # Title

# Scale features
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Build the ANN
ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=144, activation='relu'))
ann.add(tf.keras.layers.Dense(units=144, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Training ANN
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

ann.fit(X_train, y_train, batch_size=32, epochs=3000)

predictions = ann.predict(X_test)

# Create submission file
submission = pd.DataFrame()
submission['PassengerId'] = df_test_loaded.PassengerId
submission['Survived'] = predictions
submission.Survived = submission.Survived.apply(round)
submission.Survived = submission.Survived.astype(int)

submission.to_csv('data/ann_submission.csv', index=False)
