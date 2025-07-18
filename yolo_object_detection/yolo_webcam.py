from ultralytics import YOLO
import cv2

# Load trained YOLOv8 model
model = YOLO('runs/detect/train15/weights/best.pt')

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        # Resize frame to improve speed
        frame_resized = cv2.resize(frame, (640, 480))

        # Predict with confidence threshold
        results = model.predict(source=frame_resized, conf=0.5, verbose=False)

        # Draw results
        annotated_frame = results[0].plot()

        # Show result
        cv2.imshow("YOLOv8 Live Detection", annotated_frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print("❌ Error during execution:", e)

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Camera released and window closed")
