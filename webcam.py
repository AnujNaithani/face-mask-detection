import cv2 as cv
import tensorflow as tf
import numpy as np
import cv2 as cv


model = tf.keras.models.load_model("model/face_mask_model.keras")

class_names = ["with_mask", "without_mask"]


cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    img = cv.resize(frame, (128, 128))

    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    img_array = tf.keras.utils.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        label = class_names[1]
    else:
        label = class_names[0]

    predict = prediction[0][0]

    cv.putText(frame,f"{label}  {predict}", (20,50),
               cv.FONT_HERSHEY_SIMPLEX,
               1, (0,255,0), 2)

    cv.imshow("Face Mask Detection", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
