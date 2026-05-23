import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pyttsx3
import threading
import time
import os

# --- 1. SETTINGS ---
MODEL_FILE = 'action.h5'
LABEL_FILE = 'gesture_labels.npy'
SEQUENCE_LENGTH = 20
CONFIDENCE_THRESHOLD = 0.98
COOLDOWN = 2.0
STABLE_DURATION = 0.5  # seconds the gesture must be held before speaking

# --- 2. LOAD MODEL & LABELS ---
if not os.path.exists(MODEL_FILE) or not os.path.exists(LABEL_FILE):
    print("ERROR: Could not find model or label files. Run train.py first!")
    exit()

model = load_model(MODEL_FILE)
raw_labels = np.load(LABEL_FILE)
ACTIONS = np.array(sorted(set(raw_labels)))
print(f"Loaded model. Actions: {ACTIONS}")

# --- 3. TTS ---
def speak(text):
    def _speak():
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    threading.Thread(target=_speak, daemon=True).start()

# --- 4. MEDIAPIPE ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# --- 5. STATE ---
sequence = []
current_label = ''
current_confidence = 0.0
last_spoken = ''
last_speak_time = 0

stable_label = ''       # what's been consistently predicted
stable_since = 0.0      # when it first appeared stably

# --- 6. LOOP ---
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    frame_landmarks = [0.0] * 126

    if result.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            label = handedness.classification[0].label
            offset = 0 if label == "Right" else 63
            for i, lm in enumerate(hand_landmarks.landmark):
                frame_landmarks[offset + i*3]     = lm.x
                frame_landmarks[offset + i*3 + 1] = lm.y
                frame_landmarks[offset + i*3 + 2] = lm.z
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    sequence.append(frame_landmarks)
    sequence = sequence[-SEQUENCE_LENGTH:]

    if len(sequence) == SEQUENCE_LENGTH:
        input_data = np.expand_dims(sequence, axis=0).astype(np.float32)
        prediction = model.predict(input_data, verbose=0)[0]
        confidence = float(np.max(prediction))
        predicted_index = np.argmax(prediction)
        now = time.time()

        if confidence >= CONFIDENCE_THRESHOLD:
            predicted_label = ACTIONS[predicted_index]
            current_confidence = confidence

            # Reset stability timer if label changed
            if predicted_label != stable_label:
                stable_label = predicted_label
                stable_since = now

            current_label = stable_label
            held_duration = now - stable_since

            # Only speak if held stable long enough + cooldown passed
            if (held_duration >= STABLE_DURATION
                    and current_label != last_spoken
                    and (now - last_speak_time) > COOLDOWN):
                speak(current_label)
                last_spoken = current_label
                last_speak_time = now

        else:
            # Confidence dropped — reset everything
            stable_label = ''
            stable_since = 0.0
            current_label = ''
            current_confidence = 0.0
            last_spoken = ''

    # --- UI ---
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 60), (30, 30, 30), -1)

    if current_label:
        cv2.putText(frame, current_label.upper(), (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        conf_text = f"{current_confidence * 100:.1f}%"
        cv2.putText(frame, conf_text, (w - 100, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)

        # Confidence bar
        bar_w = int((w - 40) * current_confidence)
        cv2.rectangle(frame, (20, h - 20), (20 + bar_w, h - 5), (0, 255, 0), -1)

        # Stability bar (shows progress toward speaking)
        if stable_since > 0:
            held = min(time.time() - stable_since, STABLE_DURATION)
            stab_w = int((w - 40) * (held / STABLE_DURATION))
            cv2.rectangle(frame, (20, h - 35), (20 + stab_w, h - 22), (0, 165, 255), -1)
            cv2.putText(frame, "Stability", (20, h - 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

    else:
        cv2.putText(frame, "Waiting...", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)

    cv2.putText(frame, "Press 'q' to quit", (w - 160, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Sign Language Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()