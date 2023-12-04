import numpy as np
import pandas as pd
import json
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from npc_gzip.compressors.base import BaseCompressor
from npc_gzip.compressors.gzip_compressor import GZipCompressor
from npc_gzip.knn_classifier import KnnClassifier

# To pass the data from JSON file to dataframe
def pass_data():

    data = []

    with open('News_Category_Dataset_v3.json') as file:
        for line in file:
            try:
                data.append(json.loads(line))
            except:
                pass
    
    df = pd.DataFrame(data)
    df['text'] = df['headline'] + ' ' + df['short_description']
    df['label'] = df['category']

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

# Used the fit_model from the original Repository
def fit_model(
    train_text: np.ndarray, train_labels: np.ndarray, distance_metric: str = "ncd"
) -> KnnClassifier:
    """
    Fits a Knn-GZip compressor on the train
    data and returns it.

    Arguments:
        train_text (np.ndarray): Training dataset as a numpy array.
        train_labels (np.ndarray): Training labels as a numpy array.

    Returns:
        KnnClassifier: Trained Knn-Compressor model ready to make predictions.
    """

    compressor: BaseCompressor = GZipCompressor()
    model: KnnClassifier = KnnClassifier(
        compressor=compressor,
        training_inputs=train_text,
        training_labels=train_labels,
        distance_metric=distance_metric,
    )

    return model

# Used the main function from the original Repository
def main() -> None:
    print("Fetching data...")
    ((train_text, train_labels), (test_text, test_labels)) = get_data()

    print("Fitting model...")
    model = fit_model(train_text, train_labels)

    # Randomly sampling from the test set.
    # The IMDb test data comes in with all of the
    # `1` labels first, then all of the `2` labels
    # last, so we're shuffling so that our model
    # has something to predict other than `1`.

    random_indicies = np.random.choice(test_text.shape[0], 1000, replace=False)

    sample_test_text = test_text[random_indicies]
    sample_test_labels = test_labels[random_indicies]

    print("Generating predictions...")
    top_k = 3

    # Here we use the `sampling_percentage` to save time
    # at the expense of worse predictions. This
    # `sampling_percentage` selects a random % of training
    # data to compare `sample_test_text` against rather
    # than comparing it against the entire training dataset.
    (distances, labels, similar_samples) = model.predict(
        sample_test_text, top_k, sampling_percentage=0.01
    )

    print(classification_report(sample_test_labels, labels.reshape(-1)))


if __name__ == "__main__":
    main()

# Output:
#     accuracy                           0.31      1000
#    macro avg       0.23      0.17      0.17      1000
# weighted avg       0.33      0.31      0.29      1000