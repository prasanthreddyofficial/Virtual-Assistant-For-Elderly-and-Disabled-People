import streamlit as st
import cv2
import numpy as np
import math
import time
import threading
import os
import smtplib
from email.mime.text import MIMEText
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import mediapipe as mp
import speech_recognition as sr
from gtts import gTTS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configure Streamlit page
st.set_page_config(
    page_title="VIRTUAL ASSISTANT FOR THE ELDERLY AND DISABLED PEOPLE",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Access secrets via Streamlit secrets manager
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
EMAIL_USER = st.secrets["EMAIL_USER"]
EMAIL_PASSWORD = st.secrets["EMAIL_PASSWORD"]

# Custom layer if needed by your model
from tensorflow.keras.layers import DepthwiseConv2D as _DepthwiseConv2D
class CustomDepthwiseConv2D(_DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)

# CSS Styling remains the same
st.markdown("""
    <style>
    /* Global Styling */
    .stApp {
        background: linear-gradient(135deg, #1C2526 0%, #2E3738 100%);
        color: #E0E7E9;
        font-family: 'Inter', 'Segoe UI', sans-serif; 
        animation: fadeIn 1.2s ease-in-out;
    }
    /* Header and other styling rules ... (rest of your CSS) */
    </style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'sentence' not in st.session_state:
    st.session_state.sentence = []
if 'current_word' not in st.session_state:
    st.session_state.current_word = ""
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'stop_requested' not in st.session_state:
    st.session_state.stop_requested = False

def header():
    st.markdown("""
        <div class="main-header">
            <h1>VIRTUAL ASSISTANT FOR THE ELDERLY AND DISABLED PEOPLE</h1>
            <p>An AI-powered platform for seamless communication and health assistance</p>
        </div>
    """, unsafe_allow_html=True)

################################################################################
# Revised: Sign Language to Text – using st.camera_input for deployment support.
################################################################################
def sign_language_to_text():
    st.subheader("Sign Language to Text")
    st.write("Use your camera to take a picture for sign language detection.")
    # Use st.camera_input instead of cv2.VideoCapture
    img_file_buffer = st.camera_input("Take a picture")
    
    if img_file_buffer is not None:
        # Convert the image data to a format OpenCV can work with.
        file_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # Flip the image horizontally (if needed)
        img = cv2.flip(img, 1)
        
        # Initialize detector and classifier
        detector = HandDetector(maxHands=1)
        with tf.keras.utils.custom_object_scope({'DepthwiseConv2D': CustomDepthwiseConv2D}):
            # Use relative paths to load your model and labels.
            classifier = Classifier("keras_model.h5", "labels.txt")
        
        offset = 20
        imgSize = 300
        # Detect hand in the provided image
        hands_found, img_processed = detector.findHands(img, draw=True)
        if hands_found:
            hand = hands_found[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            
            if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
                aspectRatio = h / w
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                labels = [chr(i) for i in range(65, 91)]
                detected_sign = labels[index]
                st.success(f"Detected Sign: {detected_sign}")
                st.image(img, channels="BGR", caption=f"Detected Sign: {detected_sign}")
            else:
                st.warning("Hand detected but unable to process the region. Please try again.")
        else:
            st.info("No hand detected. Please try capturing another image.")

################################################################################
# (Other functions remain the same; they are assumed to work as intended.)
################################################################################
def speech_to_text():
    st.subheader("Speech to Text")
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    recognizer.pause_threshold = 1.5  
    recognizer.energy_threshold = 300  
    if st.button("Record"):
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            st.info("Listening... Speak now.")
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
        try:
            text = recognizer.recognize_google(audio)
            st.success("Recognized Text:")
            st.markdown(f"<h3 style='color:#000000;'>{text}</h3>", unsafe_allow_html=True)
        except sr.UnknownValueError:
            st.error("Sorry, I could not understand your speech.")
        except sr.RequestError as e:
            st.error(f"Request Error from Google API: {e}")

def text_to_speech():
    st.subheader("Text to Speech")
    col1, col2 = st.columns([3, 1])
    with col1:
        text = st.text_area("Enter text to convert to speech:")
    with col2:
        speak_button = st.button("Speak Now")
    if speak_button:
        if not text:
            st.warning("Please enter some text.")
        else:
            try:
                tts = gTTS(text)
                tts.save("output.mp3")
                st.audio("output.mp3", format="audio/mp3")
                st.success("Playing converted speech.")
            except Exception as e:
                st.error(f"Failed to generate audio: {e}")

@st.cache_data(ttl=600)
def get_grok_response(query):
    try:
        llm = ChatGroq(model="llama3-70b-8192", api_key=GROQ_API_KEY)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a doctor who suggests medicines based on symptoms."),
            ("human", "{query}")
        ])
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"query": query})
        return response
    except Exception as e:
        return f"Error: {e}"

def chatbot():
    st.subheader("AI Chatbot for Medicines")
    query = st.text_input("Enter your symptoms:")
    if st.button("Get Response"):
        if not query:
            st.warning("Please enter symptoms.")
        else:
            st.info("Processing your request...")
            response = get_grok_response(query)
            if response:
                st.write("AI Recommendation:", response)
            else:
                st.error("Failed to get chatbot response.")

def send_emergency_alert():
    st.subheader("Emergency Alert")
    recipient = st.text_input("Enter recipient email:")
    if st.button("Send Alert"):
        if not recipient:
            st.warning("Please enter an email address.")
            return
        if not EMAIL_USER or not EMAIL_PASSWORD:
            st.error("Email credentials not configured!")
            return
        msg = MIMEText("Emergency Alert! Immediate assistance required.")
        msg["Subject"] = "Emergency Alert"
        msg["From"] = EMAIL_USER
        msg["To"] = recipient
        try:
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(EMAIL_USER, EMAIL_PASSWORD)
                server.sendmail(EMAIL_USER, recipient, msg.as_string())
            st.success("Alert sent successfully!")
        except Exception as e:
            st.error(f"Failed to send alert: {e}")

def text_to_sign():
    st.subheader("Text to Sign Language")
    text = st.text_area("Enter text to convert to sign language:")
    if st.button("Convert to Sign Language"):
        if not text:
            st.warning("Please enter some text.")
        else:
            # Ensure that the test images are in a folder within your repository,
            # for example, "asl_alphabet_test" in the repo root.
            SIGN_IMAGE_PATH = "asl_alphabet_test"
            SIGN_LANGUAGE_DICT = {chr(i + 65): os.path.join(SIGN_IMAGE_PATH, f"{chr(i + 65)}_test.jpg") for i in range(26)}
            st.info("Starting slideshow:")
            slideshow = st.empty()
            for char in text.upper():
                if char == " ":
                    slideshow.write(" ")
                    time.sleep(1)
                elif char in SIGN_LANGUAGE_DICT:
                    image_path = SIGN_LANGUAGE_DICT[char]
                    if os.path.exists(image_path):
                        slideshow.image(image_path, caption=f"Sign for '{char}'", width=200)
                    else:
                        slideshow.warning(f"No image for '{char}'.")
                    time.sleep(1)
                else:
                    slideshow.warning(f"Unsupported character: '{char}'")
                    time.sleep(1)
            slideshow.empty()

def send_medicine_reminder_email(recipient, medicine, reminder_time):
    msg = MIMEText(f"Reminder: Time to take your medicine: {medicine}")
    msg["Subject"] = f"Medicine Reminder at {reminder_time}"
    msg["From"] = EMAIL_USER
    msg["To"] = recipient
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_USER, recipient, msg.as_string())
        st.success("Reminder email sent successfully!")
    except Exception as e:
        st.error(f"Failed to send reminder: {e}")

def schedule_medicine_reminder(recipient, medicine, reminder_time_str):
    from datetime import datetime, timedelta
    try:
        now = datetime.now()
        reminder_time = datetime.strptime(reminder_time_str, "%H:%M").replace(year=now.year, month=now.month, day=now.day)
        if reminder_time < now:
            reminder_time += timedelta(days=1)
        delay = (reminder_time - now).total_seconds()
        st.info(f"Reminder scheduled in {int(delay)} seconds.")
        timer = threading.Timer(delay, send_medicine_reminder_email, args=(recipient, medicine, reminder_time_str))
        timer.start()
    except Exception as e:
        st.error(f"Error scheduling reminder: {e}")

def medicine_reminder():
    st.subheader("Medicine Reminder")
    medicine = st.text_input("Enter medicine name:")
    recipient = st.text_input("Enter recipient email:")
    reminder_time = st.text_input("Enter reminder time (HH:MM):")
    if st.button("Schedule Reminder"):
        if not medicine or not recipient or not reminder_time:
            st.warning("Please fill in all fields.")
        else:
            schedule_medicine_reminder(recipient, medicine, reminder_time)

# Sidebar feature selection
if "selected_feature" not in st.session_state:
    st.session_state.selected_feature = "Home"

def set_feature(feature_name):
    st.session_state.selected_feature = feature_name

with st.sidebar:
    st.markdown("<div class='sidebar-header'>Features</div>", unsafe_allow_html=True)
    features = [
        ("Home", "🏠"),
        ("Sign Language to Text", "🤟"),
        ("Speech to Text", "🗣️"),
        ("Text to Speech", "🔊"),
        ("AI Chatbot for Medicines", "🤖"),
        ("Emergency Alert", "🚨"),
        ("Text to Sign Language", "👐"),
        ("Medicine Reminder", "💊")
    ]
    for feature, icon in features:
        if st.button(f"{icon} {feature}", key=f"btn_{feature.lower().replace(' ', '_')}"):
            set_feature(feature)

def main():
    header()  # Display header
    
    tab1, tab2 = st.tabs(["Dashboard", "About & Help"])
    
    with tab1:
        if st.session_state.selected_feature == "Home":
            st.title("Welcome!")
            st.write("""
            - **Sign Language to Text:** Convert hand gestures into text.
            - **Speech to Text:** Transcribe spoken words in English.
            - **Text to Speech:** Generate audio from text.
            - **AI Chatbot for Medicines:** Receive personalized medicine suggestions.
            - **Emergency Alert:** Send immediate emergency notifications.
            - **Text to Sign Language:** Convert text messages into sign language visuals.
            - **Medicine Reminder:** Schedule medication reminders via email.
            """)
        elif st.session_state.selected_feature == "Sign Language to Text":
            sign_language_to_text()
        elif st.session_state.selected_feature == "Speech to Text":
            speech_to_text()
        elif st.session_state.selected_feature == "Text to Speech":
            text_to_speech()
        elif st.session_state.selected_feature == "AI Chatbot for Medicines":
            chatbot()
        elif st.session_state.selected_feature == "Emergency Alert":
            send_emergency_alert()
        elif st.session_state.selected_feature == "Text to Sign Language":
            text_to_sign()
        elif st.session_state.selected_feature == "Medicine Reminder":
            medicine_reminder()
    
    with tab2:
        st.header("About & Help")
        st.write("""
        **Virtual Assistant for the Elderly and Disabled People** is an AI-powered platform designed to help users with communication and health-related tasks. 
        The app integrates:
        - **AI-driven sign language detection** via computer vision.
        - **Speech recognition and synthesis** for natural communication.
        - An **AI chatbot** for personalized medicine suggestions.
        - An **emergency alert system** and **medicine reminder service**.
        
        **Need Help?**
        - For technical support, please contact: support@example.com
        - Refer to the documentation in the sidebar for more details.
        """)
    
    st.markdown("""
    <hr>
    <p style='text-align: center;'>Developed by Prasanth Reddy | Version 4.5</p>
    <p style='text-align: center;'>Making communication accessible for everyone!</p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()