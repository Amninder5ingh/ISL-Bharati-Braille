import csv
import os
import cv2
import mediapipe as mp
from concurrent.futures import ProcessPoolExecutor

dataset = r"C:\C-Dac\ISL TO BHARTI BRILEIS\ISL_DATASET"

mp_hand = mp.solutions.hands


# worker initialization
def init_worker():
    global hands
    hands = mp_hand.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.7
    )


def process_image(task):

    image_path, label = task

    image = cv2.imread(image_path)

    if image is None:
        return None

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    left_hand = [0]*63
    right_hand = [0]*63

    if results.multi_hand_landmarks:

        for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness):

            temp = []

            wrist_x = hand_landmarks.landmark[0].x
            wrist_y = hand_landmarks.landmark[0].y
            wrist_z = hand_landmarks.landmark[0].z

            for lm in hand_landmarks.landmark:

                temp.append(lm.x - wrist_x)
                temp.append(lm.y - wrist_y)
                temp.append(lm.z - wrist_z)

            label_hand = handedness.classification[0].label

            if label_hand == "Left":
                left_hand = temp
            else:
                right_hand = temp

    features = left_hand + right_hand

    return features + [label]


if __name__ == "__main__":

    tasks = []

    for label in os.listdir(dataset):

        folder = os.path.join(dataset, label)

        for img in os.listdir(folder):

            path = os.path.join(folder, img)

            tasks.append((path, label))

    total = len(tasks)

    print("Total images:", total)
    print("Using 10 CPU cores")

    file = open("isl_2hands_LM.csv", "w", newline="")
    writer = csv.writer(file)

    count = 0

    with ProcessPoolExecutor(
            max_workers=10,
            initializer=init_worker) as executor:

        for result in executor.map(process_image, tasks, chunksize=50):

            if result is not None:

                writer.writerow(result)

                count += 1

                if count % 500 == 0:
                    print("Processed", count, "images")

    file.close()

    print("Dataset created successfully")