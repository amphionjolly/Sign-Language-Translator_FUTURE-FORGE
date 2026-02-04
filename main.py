import cv2
from ultralytics import YOLO
import time
import pyttsx3
import threading

# --- 1. SETUP AUDIO (TEXT-TO-SPEECH) ---
# We use a separate function to speak so it doesn't freeze the video
def speak_text(text):
    def run_speech():
        try:
            # Initialize a local engine for the thread to be safe
            engine = pyttsx3.init()
            engine.setProperty('rate', 150) # Speed: 150 is natural
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"Audio Error: {e}")

    # Run in a separate thread so the video doesn't lag/stop
    t = threading.Thread(target=run_speech)
    t.start()

# --- 2. LOAD AI MODEL ---
try:
    print("Loading AI Model...")
    model = YOLO('sign_words.pt')
    print("Model Loaded Successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- 3. SETUP WEBCAM ---
# Using Index 0 (External) as per your last working setup
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("CRITICAL ERROR: Could not access webcam.")
    exit()

# Optimization: Keep resolution low for speed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Variables for logic
frame_count = 0
skip_frames = 3       # Check AI every 3 frames
latest_results = None
current_word = ""     # What the AI currently sees
last_spoken = ""      # What we just said (to prevent repeating "Hello Hello Hello")

print("Starting Translator with Audio... Press 'q' to quit.")

while True:
    # Buffer Clearing (Anti-Lag)
    for _ in range(2):
        cap.grab()
    
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1

    # --- 4. DETECTION LOGIC ---
    if frame_count % skip_frames == 0:
        # Stream=True is faster
        latest_results = model(frame, conf=0.5, verbose=False, imgsz=320, stream=True)
        
        # Process the results
        detected_this_frame = None
        
        for r in latest_results:
            latest_results = r # Save for plotting later
            
            # Check if we found any boxes
            if len(r.boxes) > 0:
                # Find the box with the highest confidence
                best_box = max(r.boxes, key=lambda x: x.conf[0])
                class_id = int(best_box.cls[0])
                detected_this_frame = r.names[class_id]
            break
        
        # --- 5. TRANSLATOR LOGIC ---
        if detected_this_frame:
            current_word = detected_this_frame
            
            # Only speak if it's a NEW word
            if current_word != last_spoken:
                print(f"Speaking: {current_word}")
                speak_text(current_word)
                last_spoken = current_word
        else:
            # Optional: Reset if nothing is detected for a while? 
            # For now, we keep the last word on screen for stability
            pass

    # --- 6. DISPLAY LOGIC (Dual Output) ---
    display_frame = frame.copy()
    
    # A. Draw the bounding boxes (Visual AI)
    if latest_results is not None:
        display_frame = latest_results.plot()

    # B. Draw the Subtitle Bar (Text Output)
    # Black rectangle at the bottom
    h, w, _ = display_frame.shape
    cv2.rectangle(display_frame, (0, h-60), (w, h), (0, 0, 0), -1)
    
    # White Text
    text_to_show = f"Translated: {last_spoken}"
    cv2.putText(display_frame, text_to_show, (20, h-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Sign Language Translator", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()