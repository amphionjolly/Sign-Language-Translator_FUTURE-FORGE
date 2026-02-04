import cv2
import os
import time

# --- CONFIGURATION ---
# The classes you want to ADD to the model
actions = ['Fire', 'Emergency']
num_images = 50  # Number of images per sign
# Create the directory structure
SAVE_PATH = 'my_yolo_dataset'

for action in actions:
    os.makedirs(os.path.join(SAVE_PATH, action), exist_ok=True)

cap = cv2.VideoCapture(0)

print("Starting YOLO Image Collection...")
print("Instructions: Press 's' to start capturing 50 images for the displayed word.")

for action in actions:
    input(f"Ready to collect for '{action}'? Press Enter to open camera...")
    
    count = 0
    while count < num_images:
        ret, frame = cap.read()
        if not ret: break
        
        display_frame = cv2.flip(frame, 1)
        
        # UI Overlay
        cv2.putText(display_frame, f"SIGN: {action}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Collected: {count}/{num_images}", (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display_frame, "Press 'S' to save frame", (20, 450), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Data Collector", display_frame)
        
        key = cv2.waitKey(1)
        if key & 0xFF == ord('s'):
            # Save the RAW frame (no text/landmarks)
            img_name = os.path.join(SAVE_PATH, action, f"{action}_{count}.jpg")
            cv2.imwrite(img_name, frame)
            print(f"Saved {img_name}")
            count += 1
        
        if key & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

print("Collection finished! Now you must label these images using LabelImg or Roboflow.")
cap.release()
cv2.destroyAllWindows()