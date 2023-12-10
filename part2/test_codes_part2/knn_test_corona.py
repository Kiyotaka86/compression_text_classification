import pandas as pd
import re, string, demoji
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

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

def get_data(test_size=0.2):
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

# Assuming you have a function to load and split your data
(train_text, train_labels), (test_text, test_labels) = get_data()

# Text Vectorization
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_text)
X_test = vectorizer.transform(test_text)

# Convert labels to integers if they are categorical
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
