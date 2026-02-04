import cv2
import time

def test_camera_source(index, backend_name, backend_id):
    print(f"\n--- TESTING INDEX {index} with {backend_name} ---")
    
    if backend_id is None:
        cap = cv2.VideoCapture(index)
    else:
        cap = cv2.VideoCapture(index, backend_id)
    
    if not cap.isOpened():
        print(f"FAILED: Could not open handle.")
        return

    # Request specific properties to force driver negotiation
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Read actual properties
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    backend_used = cap.get(cv2.CAP_PROP_BACKEND)
    
    print(f"CONNECTED! Res: {actual_w}x{actual_h} | FPS: {actual_fps} | Backend ID: {backend_used}")

    # Read 5 frames to check for "Orange Screen" data
    for i in range(5):
        ret, frame = cap.read()
        if not ret:
            print(f"Frame {i}: READ FAILED")
        else:
            # Check the average color of the frame
            avg_color_per_row = frame.mean(axis=0)
            avg_color = avg_color_per_row.mean(axis=0)
            print(f"Frame {i}: OK | Avg Color (BGR): {avg_color}")
            
            # If standard deviation is 0, it's a solid color (Orange Screen)
            std_dev = frame.std()
            if std_dev < 1.0:
                print(">>> WARNING: SOLID COLOR DETECTED (Orange/Black Screen) <<<")
            else:
                print(">>> OK: Frame has detail (Not solid color) <<<")

    cap.release()

# Test standard backends
test_camera_source(0, "DEFAULT (MSMF)", cv2.CAP_MSMF)
test_camera_source(0, "DIRECTSHOW (DSHOW)", cv2.CAP_DSHOW)
# If you have an external cam, test index 1
test_camera_source(1, "DEFAULT (Index 1)", None)