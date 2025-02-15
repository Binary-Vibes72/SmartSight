### Import Section Starts here
from ultralytics import YOLO
import cv2
import speech_recognition as sr
from gtts import gTTS
import os
import openai
import threading
import torch

# Set OpenAI API key
openai.api_key = os.environ.get("OPEN_KEY")

### Function Definitions
def listen_to_user():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I could not understand.")
            return None

def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")
    os.system("mpg321 response.mp3")

def get_ai_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=50
    )
    return response.choices[0].text.strip()

### Modeling starts here

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # Use the nano model for better FPS

# Check for GPU and move the model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)  # Move model to GPU
print(f"Using device: {device}")

def object_detection():
    cap = cv2.VideoCapture(0)  # '0' for default webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        results = model.predict(source=frame, imgsz=320, half=True)  # Lower resolution for faster inference

        # Display the frame with detections
        annotated_frame = results[0].plot()  # Draw bounding boxes and labels
        cv2.imshow("Object Detection", annotated_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def voice_assistant():
    while True:
        query = listen_to_user()
        if query:
            # Generate AI response based on the query
            response = get_ai_response(query)
            print(f"AI: {response}")
            speak(response)

# Start threads
threading.Thread(target=object_detection).start()
threading.Thread(target=voice_assistant).start()