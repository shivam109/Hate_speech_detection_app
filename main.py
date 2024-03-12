# Import necessary modules
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
import requests

# Initialize NLTK stemmer and stopwords
stemmer = nltk.SnowballStemmer("english")
stopword = set(stopwords.words('english'))

# Function to clean text data
def clean(text):

    # Convert text to lowercase
    text = str(text).lower()

    # Remove square brackets and their contents
    text = re.sub('\[.*?\]', '', text)

    # Remove URLs
    text = re.sub('https?://\S+|www\.\S+', '', text)

    # Remove HTML tags
    text = re.sub('<.*?>+', '', text)

    # Remove punctuation
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    # Remove newline characters
    text = re.sub('\n', '', text)

    # Remove words containing digits
    text = re.sub('\w*\d\w*', '', text)

    # Remove stopwords
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)

    # Apply stemming
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text

# Function to call OCR.space API
def extract_text(image_file):
    api_key = 'K89461099688957'
    endpoint = 'https://api.ocr.space/parse/image'
    payload = {
        'apikey': api_key,
        'language': 'eng',
    }
    with open(image_file, 'rb') as file:
        result = requests.post(endpoint, files={image_file: file}, data=payload)
        return result.json()

# Load your data from CSV file
data = pd.read_csv("twitter.csv")

# Map numerical labels to descriptive categories
data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive Language", 2: "No Hate and Offensive"})

# Keep only relevant columns
data = data[["tweet", "labels"]]

# Apply the cleaning function to the "tweet" column
data["tweet"] = data["tweet"].apply(clean)

# Split the data into features (x) and labels (y)
x = np.array(data["tweet"])
y = np.array(data["labels"])

# Initialize CountVectorizer for feature extraction
cv = CountVectorizer()

# Transform text data into a bag-of-words representation
X = cv.fit_transform(x)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Initialize a Decision Tree Classifier
clf = DecisionTreeClassifier()

# Train the classifier on the training data
clf.fit(X_train, y_train)