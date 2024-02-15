import pickle
import cv2
import mediapipe as mp
import numpy as np
import warnings

# Suppress the deprecation warning
warnings.filterwarnings("ignore")

# Load models
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

model_dict_two = pickle.load(open('./model_two.p', 'rb'))
model_two = model_dict_two['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'Hello',
               1: 'Thank You', 
               2: 'Yes', 
               3: 'No', 
               4: 'I Love You', 
               5: 'My Name Is', 
               6: 'Mohamed Hazem',
               7: 'GoodBye',
               8: 'This is',
               9: 'Sign Language Translator',
               10 : 'I am Sorry',
               11 : 'Help Me',
               12 : 'I Want To Drink',
               13 : 'I Want To Eat',
               14 : 'I am Fine',
               15 : 'I am Not Fine',

}

while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Check the length of data_aux
        if len(data_aux) == 42:
            features_for_prediction = np.asarray(data_aux)
            prediction = model.predict([features_for_prediction])
        elif len(data_aux) == 84:
            features_for_prediction = np.asarray(data_aux)
            prediction = model_two.predict([features_for_prediction])
        else:
            print("Unexpected length of data_aux. Handling accordingly...")
            features_for_prediction = np.asarray(data_aux)[:42]
            prediction = model.predict([features_for_prediction])

        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (int(min(x_) * W) - 10, int(min(y_) * H) - 10),
                      (int(max(x_) * W) - 10, int(max(y_) * H) - 10), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (int(min(x_) * W) - 10, int(min(y_) * H) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # 27 is the ASCII code for the 'Esc' key
        break

cap.release()
cv2.destroyAllWindows()
