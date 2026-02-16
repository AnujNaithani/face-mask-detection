import cv2 as cv
import tensorflow as tf
import numpy as np
import cv2 as cv
# Load your trained model
model = tf.keras.models.load_model("face_mask_model_with_augmentation.keras")

# Class names (change according to your dataset)
class_names = ["with_mask", "without_mask"]

# Start webcam
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Resize frame to match model input
    img = cv.resize(frame, (128, 128))

    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Convert to array
    img_array = tf.keras.utils.img_to_array(img)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Predict
    prediction = model.predict(img_array)

    # For sigmoid output
    if prediction[0][0] > 0.5:
        label = class_names[1]
    else:
        label = class_names[0]

    predict = prediction[0][0]
    # Show label on webcam frame
    cv.putText(frame,f"{label}  {predict}", (20,50),
               cv.FONT_HERSHEY_SIMPLEX,
               1, (0,255,0), 2)
    # Display webcam
    cv.imshow("Face Mask Detection", frame)

    # Press q to exit
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
