import numpy as np
import pandas as pd
import json
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# To pass the data from JSON file to dataframe
def pass_data():

    df = pd.read_csv('train_40k.csv', encoding='latin-1')

    # reduce the sample size due to heavy computation
    # sample_size = 20000
    # # Sample the dataframe if it is larger than the sample size
    # if len(df) > sample_size:
    #     df = df.sample(n=sample_size, random_state=42)


    df['text'] = df['Text']
    df['label'] = df['Cat1']

    return df

# To split the data into train and test
def get_data(test_size=0.2):
    df = pass_data()

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=test_size, random_state=42)

    # Convert to numpy arrays
    label_encoder = LabelEncoder()

    train_text = X_train.to_numpy()
    train_labels = label_encoder.fit_transform(y_train.to_numpy())
    test_text = X_test.to_numpy()
    test_labels = label_encoder.fit_transform(y_test.to_numpy())

    train = (train_text, train_labels)
    test = (test_text, test_labels)

    return (train, test)


# extract the train test data
(train_text, train_labels), (test_text, test_labels) = get_data()

# Text Vectorization
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_text)
X_test = vectorizer.transform(test_text)

# Convert labels to integers
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# Train the standard KNN model
standard_knn_model = KNeighborsClassifier()
standard_knn_model.fit(X_train, train_labels_encoded)

# Evaluate the standard KNN model
standard_predictions = standard_knn_model.predict(X_test)
print("Standard KNN Model Performance:")
print(classification_report(test_labels_encoded, standard_predictions))

# Output:
#     accuracy                           0.75      8000
#    macro avg       0.74      0.76      0.74      8000
# weighted avg       0.76      0.75      0.75      8000