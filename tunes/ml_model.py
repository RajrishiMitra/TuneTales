import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
import googleapiclient.discovery
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression  # Import Logistic Regression classifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from scikitplot.metrics import plot_confusion_matrix
#from wordcloud import WordCloud
#nltk.download('stopwords')
#nltk.download('wordnet')
label_mapping = {"surprise": 1, "love": 2, "joy": 3, "fear": 4, "anger": 5, "sadness": 6}

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'tunes', 'static', 'dataset')

df_train = pd.read_csv(os.path.join(DATASET_DIR, 'train.txt'), delimiter=';', names=['text', 'label'])
df_val = pd.read_csv(os.path.join(DATASET_DIR, 'val.txt'), delimiter=';', names=['text', 'label'])
df = pd.concat([df_train, df_val])
df.reset_index(inplace=True, drop=True)

test_df = pd.read_csv(os.path.join(DATASET_DIR, 'test.txt'), delimiter=';', names=['text', 'label'])


df = pd.concat([df_train, df_val])
df.reset_index(inplace=True, drop=True)
def text_transformation(df_col):
    corpus = []
    for item in df_col:
        new_item = re.sub('[^a-zA-Z]', ' ', str(item))
        new_item = new_item.lower()
        new_item = new_item.split()
        new_item = [lm.lemmatize(word) for word in new_item if word not in set(stopwords.words('english'))]
        corpus.append(' '.join(str(x) for x in new_item))
    return corpus
def custom_encoder(df):
    df['label'] = df['label'].map(label_mapping)
lm = WordNetLemmatizer()
# Preprocess data
custom_encoder(df)
corpus = text_transformation(df['text'])
# Create CountVectorizer
cv = CountVectorizer(ngram_range=(1, 2))
traindata = cv.fit_transform(corpus)
X = traindata
y = df['label']
logistic_regression = LogisticRegression(max_iter=1000)  # You can adjust max_iter as needed
logistic_regression.fit(X, y)
test_df['label'] = test_df['label'].map(label_mapping)  # Update labels in the test set
X_test, y_test = test_df['text'], test_df['label']
custom_encoder(test_df)
test_corpus = text_transformation(X_test)
testdata = cv.transform(test_corpus)
# Predictions using the Logistic Regression classifier
predictions = logistic_regression.predict(testdata)

# Calculate evaluation metrics
acc_score = accuracy_score(y_test, predictions)
pre_score = precision_score(y_test, predictions, average='weighted')
rec_score = recall_score(y_test, predictions, average='weighted')
confusion = confusion_matrix(y_test, predictions)
# YouTube API Initialization
api_key = 'AIzaSyDZVL27q4f5kND3Z34iNRR7Vcx0qFbG97M'
youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey=api_key)

def expression_check(prediction_input):
    mood = ''
    if prediction_input == 1:
        mood = 'Surprise'
    elif prediction_input == 2:
        mood = 'Love'
    elif prediction_input == 3:
        mood = 'Pop'
    elif prediction_input == 4:
        mood = 'Fear'
    elif prediction_input == 5:
        mood = 'Rock'
    elif prediction_input == 6:
        mood = 'Sad'
    else:
        mood = 'Neutral'

    search_query = f"{mood} music"
    max_results = 10

    search_response = youtube.search().list(
        q=search_query,
        type='video',
        part='id',
        maxResults=max_results
    ).execute()

    video_ids = [item['id']['videoId'] for item in search_response['items']]
    
    video_urls = [f'https://www.youtube.com/watch?v={video_id}' for video_id in video_ids]
    return mood, video_urls
def sentiment_predictor(input):
    input = text_transformation(input)
    transformed_input = cv.transform(input)
    prediction = logistic_regression.predict(transformed_input)
    mood, video_urls = expression_check(prediction)
    
    # Print the detected mood and recommended video URLs
    print("Detected Mood:", mood)
    print("Recommended Music:")
    for idx, video_url in enumerate(video_urls, 1):
        print(f"{idx}. {video_url}")

    return {'mood': mood, 'video_urls': video_urls}
