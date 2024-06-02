import pandas as pd
import re
import nltk
import googleapiclient.discovery
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import requests

# Uncomment the following lines if you haven't downloaded NLTK data
# nltk.download('stopwords')
# nltk.download('wordnet')

# Mapping for label names to numerical values
label_mapping = {"surprise": 1, "love": 2, "joy": 3, "fear": 4, "anger": 5, "sadness": 6}

# Set up file paths
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'tunes', 'static', 'dataset')

# Load training and validation data
df_train = pd.read_csv(os.path.join(DATASET_DIR, 'train.txt'), delimiter=';', names=['text', 'label'])
df_val = pd.read_csv(os.path.join(DATASET_DIR, 'val.txt'), delimiter=';', names=['text', 'label'])
df = pd.concat([df_train, df_val])
df.reset_index(inplace=True, drop=True)

# Load test data
test_df = pd.read_csv(os.path.join(DATASET_DIR, 'test.txt'), delimiter=';', names=['text', 'label'])

# Function to preprocess text data
def text_transformation(df_col):
    corpus = []
    for item in df_col:
        # Remove non-alphabetic characters
        new_item = re.sub('[^a-zA-Z]', ' ', str(item))
        new_item = new_item.lower()  # Convert text to lowercase
        new_item = new_item.split()  # Tokenize text
        # Lemmatize words and remove stopwords
        new_item = [lm.lemmatize(word) for word in new_item if word not in set(stopwords.words('english'))]
        corpus.append(' '.join(str(x) for x in new_item))
    return corpus

# Function to map labels to numerical values
def custom_encoder(df):
    df['label'] = df['label'].map(label_mapping)

# Initialize WordNet Lemmatizer
lm = WordNetLemmatizer()

# Preprocess training and validation data
custom_encoder(df)
corpus = text_transformation(df['text'])

# Create CountVectorizer
cv = CountVectorizer(ngram_range=(1, 2))
traindata = cv.fit_transform(corpus)
X = traindata
y = df['label']

# Train Logistic Regression classifier
logistic_regression = LogisticRegression(max_iter=1000)
logistic_regression.fit(X, y)

# Preprocess test data
test_df['label'] = test_df['label'].map(label_mapping)
X_test, y_test = test_df['text'], test_df['label']
custom_encoder(test_df)
test_corpus = text_transformation(X_test)
testdata = cv.transform(test_corpus)

# Predictions using the Logistic Regression classifier
predictions = logistic_regression.predict(testdata)

# Initialize YouTube API
api_key = 'AIzaSyBSuT0OIf-vSRf0MDpFcYnK6LnvfdJ3Zhs'
youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey=api_key)

# Function to search YouTube for music based on predicted mood
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

# Function to fetch book recommendations based on the given category
def fetch_books(category):
    try:
        # Perform book search using Google Books API
        url = f'https://www.googleapis.com/books/v1/volumes?q=subject:{category}&maxResults=5'
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if 'items' in data:
            return data['items']
    except requests.RequestException as e:
        print(f"Error fetching books: {e}")
        return None

def sentiment_predictor(input):
    input = text_transformation(input)
    transformed_input = cv.transform(input)
    prediction = logistic_regression.predict(transformed_input)
    mood, video_urls = expression_check(prediction)
    
    # Fetch book recommendations based on the predicted mood
    category_mapping = {1: "comedy", 2: "romantic", 3: "adventure", 4: "inspirational", 5: "inspirational", 6: "romantic"}
    category = category_mapping.get(prediction[0], "fiction")
    books = fetch_books(category)
    
    return {'mood': mood, 'video_urls': video_urls, 'books': books}


def cam_sentiment_predictor(prediction):
    mood, video_urls = expression_check(prediction)
    category_mapping = {1: "comedy", 2: "romantic", 3: "adventure", 4: "inspirational", 5: "inspirational", 6: "romantic"}
    category = category_mapping.get(prediction, "fiction")
    books = fetch_books(category)
    
    return {'mood': mood, 'video_urls': video_urls, 'books': books}