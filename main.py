import streamlit as st
import cv2
import numpy as np
import math
import time
import threading
import os
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import mediapipe as mp
import speech_recognition as sr
from gtts import gTTS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(
    page_title="VIRTUAL ASSISTANT FOR THE ELDERLY AND DISABLED PEOPLE",
    layout="wide",
    initial_sidebar_state="expanded",
)

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
EMAIL_USER = st.secrets["EMAIL_USER"]
EMAIL_PASSWORD = st.secrets["EMAIL_PASSWORD"]


from tensorflow.keras.layers import DepthwiseConv2D as _DepthwiseConv2D

class CustomDepthwiseConv2D(_DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)

st.markdown("""
    <style>
    /* Global Styling */
    .stApp {
        background: linear-gradient(135deg, #1C2526 0%, #2E3738 100%);
        color: #E0E7E9; /* Soft white text */
        font-family: 'Inter', 'Segoe UI', sans-serif; 
        animation: fadeIn 1.2s ease-in-out; 
    }

    /* Header */
    .main-header {
        background: linear-gradient(90deg, #2E3738, #404C4D); 
        padding: 2.5rem;
        border-radius: 12px;
        color: #FFFFFF;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        animation: slideDown 0.8s ease-out; 
    }
    .main-header h1 {
        font-size: 2.6rem;
        font-weight: 600;
        margin: 0;
        letter-spacing: 0.5px;
    }
    .main-header p {
        font-size: 1.2rem;
        margin-top: 0.75rem;
        color: #A8B5B5; 
        font-weight: 300;
        animation: fadeInText 1.5s ease-in;
    }

    /* Sidebar */
    .css-1d391kg {
        background: #2E3738 !important; 
        border-right: 1px solid #404C4D;
    }
    .sidebar-header {
        padding: 1.5rem;
        font-size: 1.5rem;
        font-weight: 500;
        color: #E0E7E9;
        border-bottom: 1px solid #4A5556;
        background: linear-gradient(45deg, #2E3738, #404C4D);
        border-radius: 8px 8px 0 0;
    }
    .css-1d391kg .stButton>button {
        background: #5A6768; 
        color: #FFFFFF;
        border-radius: 8px;
        padding: 0.7rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease; 
    }
    .css-1d391kg .stButton>button:hover {
        background: #4A5556; 
        transform: translateY(-2px); 
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    }

    /* Main Buttons */
    .stButton>button {
        background: linear-gradient(45deg, #5A6768, #6E7B7C); 
        color: #FFFFFF;
        border-radius: 8px;
        padding: 0.7rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px); /* Lift effect */
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        animation: subtlePulse 1.5s infinite; 
    }

    /* Containers */
    .video-container, .stContainer {
        background: #2E3738; 
        padding: 1.75rem;
        border-radius: 10px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
        color: #E0E7E9;
        border: 1px solid #404C4D;
        transition: all 0.3s ease;
        animation: fadeUp 1s ease-out; 
    }
    .video-container:hover, .stContainer:hover {
        transform: translateY(-4px); 
        border-color: #6E7B7C; /* Light gray accent */
    }

    /* Inputs & Text Areas */
    textarea, input {
        background: #374242 !important; 
        color: #E0E7E9 !important;
        border: 1px solid #4A5556 !important;
        border-radius: 6px;
        padding: 0.75rem;
        transition: all 0.3s ease;
    }
    textarea:focus, input:focus {
        border-color: #6E7B7C !important;
        box-shadow: 0 0 8px rgba(110, 123, 124, 0.3);
        outline: none;
    }

    /* Tabs */
    .st-c4, .st-c5 {
        background: #2E3738 !important;
        color: #E0E7E9 !important;
        border-radius: 8px;
    }
    .st-c4 button[aria-selected="true"], .st-c5 button[aria-selected="true"] {
        background: linear-gradient(45deg, #5A6768, #6E7B7C) !important;
        color: #FFFFFF !important;
        font-weight: 500;
        animation: fadeIn 0.5s ease-in;
    }

    /* Links */
    a {
        color: #A8B5B5; 
        text-decoration: none;
        transition: all 0.3s ease;
    }
    a:hover {
        color: #E0E7E9; 
        text-shadow: 0 0 4px rgba(224, 231, 233, 0.4);
    }

    /* Headings & Emphasis */
    h2, h3, .stHeader {
        color: #E0E7E9 !important;
        font-weight: 500;
    }
    h2 {
        font-size: 2rem;
        animation: fadeInText 1.2s ease-in;
    }
    h3 {
        font-size: 1.5rem;
    }
    strong, b {
        color: #FFFFFF;
        font-weight: 600;
    }

    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes slideDown {
        from { transform: translateY(-15px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    @keyframes fadeUp {
        from { transform: translateY(10px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    @keyframes fadeInText {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes subtlePulse {
        0%, 100% { transform: translateY(-2px); }
        50% { transform: translateY(-4px); }
    }

  
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #2E3738;
    }
    ::-webkit-scrollbar-thumb {
        background: #5A6768;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #6E7B7C;
    }
</style>
""", unsafe_allow_html=True)

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

def sign_language_to_text():
    st.subheader("Sign Language to Text")
    st.write("Detecting sign language using AI, OpenCV, and cvzone...")
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    with tf.keras.utils.custom_object_scope({'DepthwiseConv2D': CustomDepthwiseConv2D}):
        classifier = Classifier(r"D:\AI Assitant\keras_model.h5", r"D:\AI Assitant\labels.txt")
    
    offset = 20
    imgSize = 300
    word_creation_time = None
    labels = [chr(i) for i in range(65, 91)]  # A-Z

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="video-container"><h2>Sign Language Detection</h2></div>', unsafe_allow_html=True)
        stframe = st.empty()
    with col2:
        current_word_display = st.empty()
        sentence_display = st.empty()
    
    if not st.session_state.processing:
        if st.button("Start Translation", key="start_button"):
            st.session_state.processing = True
            st.session_state.stop_requested = False
    else:
        if st.button("Stop", key="stop_button"):
            st.session_state.stop_requested = True

    while not st.session_state.stop_requested and st.session_state.processing:
        success, img = cap.read()
        if not success:
            break
        img = cv2.flip(img, 1)
        imgOutput = img.copy()
        hands_found, img = detector.findHands(img, draw=False)

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

                if word_creation_time is None:
                    word_creation_time = time.time()
                if time.time() - word_creation_time > 3:
                    st.session_state.current_word += labels[index]
                    word_creation_time = None
        else:
            if word_creation_time is not None:
                if st.session_state.current_word:
                    st.session_state.sentence.append(st.session_state.current_word)
                    st.session_state.current_word = ""
                else:
                    st.session_state.sentence.append('')
                word_creation_time = None

        cv2.putText(imgOutput, st.session_state.current_word, (10, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        current_word_display.markdown(f"""
            <div class="current-word">
                <strong>Current Word:</strong> {st.session_state.current_word}
            </div>
        """, unsafe_allow_html=True)

        sentence_display.markdown(f"""
            <div class="sentence-display">
                <h3>Translated Sentence:</h3>
                <p>{' '.join(st.session_state.sentence)}</p>
            </div>
        """, unsafe_allow_html=True)

        stframe.image(imgOutput, channels="BGR", use_container_width=True)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if st.session_state.stop_requested:
        final_sentence = " ".join(st.session_state.sentence)
        st.success(f"Final Sentence: {final_sentence}")
        st.session_state.processing = False
        st.session_state.stop_requested = False
        cap.release()
        cv2.destroyAllWindows()

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
            SIGN_IMAGE_PATH = r"D:\AI Assitant\asl_dataset\asl_alphabet_test"
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
    header()  # Display professional header
    
    tab1, tab2 = st.tabs(["Dashboard", "About & Help"])
    
    with tab1:
        if st.session_state.selected_feature == "Home":
            st.title("Welcome!")
            st.write("Explore the features of this AI-powered virtual assistant.")
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
    <p style='text-align: center;'>Developed by Prasanth Reddy & Team | Version 4.5</p>
    <p style='text-align: center;'>Making communication accessible for everyone!</p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
