# Import necessary modules
import streamlit as st
import pyttsx3
import threading
import tweepy
from main import clean, clf, cv, extract_text


# Twitter API credentials
consumer_key = 'gKb0wDrMVEoFqM3Hz0M3jlNc2'
consumer_secret = 'KjmBdYKhdwGSrhEqnUU4x12W4yUviZVIWg6LMP3XZolBfCUrsZ'
access_token = '771677148713144320-zC025J7bzFH8MzQkLEiuHQXSqu7mQaM'
access_token_secret = 'fsyNWt90MkMlwkjxD6zKfEUZAFLoxrKVtpiYMsBXRy5a0'

# Set up OAuthHandler
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

# Set access token
auth.set_access_token(access_token, access_token_secret)

# Create a Tweepy API object
api = tweepy.API(auth)

# Streamlit app title
st.title("Hate Speech Detection App")

# Input for choice
input_choice = st.radio("Select Input Type:", ["Text Input", "Twitter URL", "Upload Image"])

# Function to speak text asynchronously in a separate thread
def speak_async_thread(text):
    thread = threading.Thread(target=speak_async, args=(text,))
    thread.start()

# Function to speak text asynchronously
def speak_async(text):

    # Reinitialize the text-to-speech engine each time
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def find_verdict(user_input):

    # Clean the user input
    cleaned_input = clean(user_input)

    # Transform the input using the CountVectorizer
    input_data = cv.transform([cleaned_input]).toarray()

    # Predict using the trained model
    verdict = clf.predict(input_data)[0]

    # Display the result
    st.success(f"Verdict: {verdict}")

    # Convert the verdict to text for speech
    speech_text = f"The Verdict is {verdict}"

    # Use text-to-speech to say the result asynchronously in a separate thread
    speak_async_thread(speech_text)

# For text input
if input_choice == 'Text Input':
    user_input = st.text_area("Enter your text:")
    if st.button("Detect Hate Speech"):
        if user_input:

            #  Find result based on the text retrieved
            find_verdict(user_input)

        else:

        # Display a warning if no text is entered
            st.warning("Please enter text before clicking the button.")

# For twitter URL
elif input_choice == 'Twitter URL':
    twitter_url = st.text_input("Enter Twitter URL:", "")
    if st.button("Detect Hate Speech"):
        if twitter_url:
            try:

                # Extract tweet content using tweepy
                tweet_id = twitter_url.split("/")[-1]
                tweet = api.get_status(tweet_id, tweet_mode="extended")
                tweet_content = tweet.full_text

                find_verdict(tweet_content)

            # Use TweepyException from tweepy.errors
            except tweepy.errors.TweepyException as e:  
                st.error(f"Error fetching tweet: {str(e)}")
        else:

            # Display a warning if no text is entered
            st.warning("Please enter URL before clicking the button.")

# For image input
elif input_choice == 'Upload Image':

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:

        # Display the uploaded image in the Streamlit app
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        if st.button('Detect Hate Speech'):

            # Extract text from the uploaded image using OCR.space API
            extracted_text = extract_text(uploaded_file.name)

            # Access the parsed results from the OCR API response and retrieve text content
            user_input = extracted_text['ParsedResults'][0]['ParsedText']
            find_verdict(user_input)