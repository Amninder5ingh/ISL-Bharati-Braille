import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque, Counter
import warnings

warnings.filterwarnings("ignore")

model = joblib.load("isl_2hand_model.pkl")
scaler = joblib.load("isl_2hand_scaler.pkl")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

prediction_history = deque(maxlen=20)

print("Camera ready...")

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    frame = cv2.flip(frame,1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    left_hand = [0]*63
    right_hand = [0]*63

    if results.multi_hand_landmarks:

        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness):

            temp = []

            # wrist position
            wrist_x = hand_landmarks.landmark[0].x
            wrist_y = hand_landmarks.landmark[0].y
            wrist_z = hand_landmarks.landmark[0].z

            for lm in hand_landmarks.landmark:

                # wrist normalization
                temp.append(lm.x - wrist_x)
                temp.append(lm.y - wrist_y)
                temp.append(lm.z - wrist_z)

            hand_label = handedness.classification[0].label

            if hand_label == "Left":
                left_hand = temp
            else:
                right_hand = temp

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    features = left_hand + right_hand

    if sum(features) != 0:

        features = np.array(features).reshape(1,-1)

        features = scaler.transform(features)

        prediction = model.predict(features)[0]

        prediction_history.append(prediction)

        final_prediction = Counter(prediction_history).most_common(1)[0][0]

        cv2.rectangle(frame,(30,30),(300,120),(0,0,0),-1)

        cv2.putText(frame,
                    str(final_prediction),
                    (50,100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0,255,0),
                    3)

    cv2.imshow("ISL Two Hand Detection",frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()