import cv2
from ultralytics import YOLO
import time

# 1. Load the model
try:
    print("Loading AI Model...")
    # Using the .pt model for detection
    model = YOLO('sign_words.pt')
    print("Model Loaded Successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# 2. Setup Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("CRITICAL ERROR: Could not access webcam.")
    exit()

# Optimization: Set a smaller resolution for faster AI processing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Performance variables
frame_count = 0
skip_frames = 3  # Run AI every 3rd frame. Increase to 5 if still laggy.
latest_results = None

print("Starting Optimized Prototype... Press 'q' to quit.")

while True:
    # BUFFER CLEARING: Read multiple frames but only keep the last one 
    # to ensure real-time performance on slower CPUs
    for _ in range(2):
        cap.grab()
    
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1

    # 3. Detect signs - Only run every 'skip_frames' to save CPU
    if frame_count % skip_frames == 0:
        # imgsz=320 reduces the image size inside the AI for 2x speed boost
        # stream=True handles memory better for continuous video
        latest_results = model(frame, conf=0.4, verbose=False, imgsz=320, stream=True)
        
        # Extract the first result from the stream generator
        for r in latest_results:
            latest_results = r
            break

    # 4. Display Logic
    if latest_results is not None:
        # Show annotated frame from the last successful detection
        annotated_frame = latest_results.plot()
        cv2.imshow("Sign Language Translator", annotated_frame)
    else:
        # If no AI results yet, just show the raw frame
        cv2.imshow("Sign Language Translator", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

#ok faster than before, now you go back to the first messages of cristovae project we are making, do you remember dual output- text on screen and audio outpu through speakers. from this same code we can start developing it(plese dont destroy the current working system).