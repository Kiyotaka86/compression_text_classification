import numpy as np
import pandas as pd
import re, string, demoji
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from npc_gzip.compressors.base import BaseCompressor
from npc_gzip.compressors.gzip_compressor import GZipCompressor
from npc_gzip.knn_classifier import KnnClassifier

# Deep Cleaning methods from https://www.kaggle.com/code/ludovicocuoghi/twitter-sentiment-analysis-with-bert-vs-roberta
def strip_emoji(text):
    return demoji.replace(text, '') #remove emoji

#Remove punctuations, links, mentions and \r\n new line characters
def strip_all_entities(text): 
    text = text.replace('\r', '').replace('\n', ' ').replace('\n', ' ').lower() #remove \n and \r and lowercase
    text = re.sub(r"(?:\@|https?\://)\S+", "", text) #remove links and mentions
    text = re.sub(r'[^\x00-\x7f]',r'', text) #remove non utf8/ascii characters such as '\x9a\x91\x97\x9a\x97'
    banned_list= string.punctuation + 'Ã'+'±'+'ã'+'¼'+'â'+'»'+'§'
    table = str.maketrans('', '', banned_list)
    text = text.translate(table)
    return text

#clean hashtags at the end of the sentence, and keep those in the middle of the sentence by removing just the # symbol
def clean_hashtags(tweet):
    new_tweet = " ".join(word.strip() for word in re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', tweet)) #remove last hashtags
    new_tweet2 = " ".join(word.strip() for word in re.split('#|_', new_tweet)) #remove hashtags symbol from words in the middle of the sentence
    return new_tweet2

#Filter special characters such as & and $ present in some words
def filter_chars(a):
    sent = []
    for word in a.split(' '):
        if ('$' in word) | ('&' in word):
            sent.append('')
        else:
            sent.append(word)
    return ' '.join(sent)

def remove_mult_spaces(text): # remove multiple spaces
    return re.sub("\s\s+" , " ", text)

def pass_data():

    train_df = pd.read_csv('Corona_NLP_train.csv', encoding='latin-1')

    train_text = []
    for t in train_df.OriginalTweet:
        train_text.append(remove_mult_spaces(filter_chars(clean_hashtags(strip_all_entities(strip_emoji(t))))))

    train_df['text'] = train_text
    train_df['label'] = train_df['Sentiment']
    
    test_df = pd.read_csv('Corona_NLP_test.csv', encoding='latin-1')

    test_text = []
    for t in test_df.OriginalTweet:
        test_text.append(remove_mult_spaces(filter_chars(clean_hashtags(strip_all_entities(strip_emoji(t))))))
    
    test_df['text'] = test_text
    test_df['label'] = test_df['Sentiment']

    return test_df, train_df

def get_data():
    train_df, test_df = pass_data()

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_df['text'], test_df['text'], train_df['label'], test_df['label']  

    # Convert to numpy arrays
    label_encoder = LabelEncoder()

    train_text = X_train.to_numpy()
    train_labels = label_encoder.fit_transform(y_train.to_numpy())
    test_text = X_test.to_numpy()
    test_labels = label_encoder.fit_transform(y_test.to_numpy())

    train = (train_text, train_labels)
    test = (test_text, test_labels)

    return (train, test)


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
    top_k = 5

    # Here we use the `sampling_percentage` to save time
    # at the expense of worse predictions. This
    # `sampling_percentage` selects a random % of training
    # data to compare `sample_test_text` against rather
    # than comparing it against the entire training dataset.
    (distances, labels, similar_samples) = model.predict(
        sample_test_text, top_k, sampling_percentage=0.1
    )

    print(classification_report(sample_test_labels, labels.reshape(-1)))


if __name__ == "__main__":
    main()