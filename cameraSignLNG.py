import os
import sys
import time
import cv2
import pickle
import mediapipe as mp
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Ρυθμίσεις Διαδρομών ---
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
AI_MODEL_PATH = os.path.join(BASE_DIR, "sign_language_model.pkl")

if not os.path.exists(MODEL_PATH) or not os.path.exists(AI_MODEL_PATH):
    print("Σφάλμα: Δεν βρέθηκαν τα αρχεία του μοντέλου.")
    sys.exit()

# Φόρτωση του εκπαιδευμένου AI Μοντέλου μας
with open(AI_MODEL_PATH, 'rb') as f:
    clf_model = pickle.load(f)

# Συνδέσεις για να ζωγραφίζουμε το χέρι
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), 
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), 
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
]

def draw_hand(image_bgr, hand_landmarks):
    h, w = image_bgr.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(image_bgr, pts[a], pts[b], (0, 255, 0), 2)
    for x, y in pts:
        cv2.circle(image_bgr, (x, y), 4, (0, 0, 255), -1)

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # --- ΜΕΤΑΒΛΗΤΕΣ ΓΙΑ ΤΟΝ ΣΧΗΜΑΤΙΣΜΟ ΛΕΞΕΩΝ ---
    current_word = ""           # Η λέξη που φτιάχνει
    previous_prediction = ""    # Τι έβλεπε στο προηγούμενο καρέ
    frame_counter = 0           # Μετρητής σταθερότητας
    FRAMES_TO_CONFIRM = 30      # Καρέ αναμονής (15 καρέ δλδ μισό δευτερόλεπτο)

    feature_names = []
    for i in range(21):
        feature_names.extend([f'x{i}', f'y{i}', f'z{i}'])

    print("\n" + "="*50)
    print("Η κάμερα ανοίγει! ΟΔΗΓΙΕΣ ΧΡΗΣΗΣ:")
    print("1. Κράτα το χέρι σταθερό για μισό δευτερόλεπτο για να πληκτρολογηθεί το γράμμα.")
    print("2. Πάτα το 'c' στο πληκτρολόγιο για να καθαρίσεις/σβήσεις όλη τη λέξη.")
    print("3. Πάτα το 'q' για έξοδο.")
    print("="*50 + "\n")
    
    start = time.time()

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_bgr = cv2.flip(frame_bgr, 1)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int((time.time() - start) * 1000)
            
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                hand_lms = result.hand_landmarks[0]
                draw_hand(frame_bgr, hand_lms)

        
                row = []
                for lm in hand_lms:
                    row.extend([lm.x, lm.y, lm.z])
                
                X_live = pd.DataFrame([row], columns=feature_names)
                prediction = clf_model.predict(X_live)[0]
                
                # --- ΛΟΓΙΚΗ WORD BUILDER ---
                if prediction == previous_prediction:
                    frame_counter += 1
                else:
                    frame_counter = 0
                    previous_prediction = prediction

                # Αν το χέρι έμεινε σταθερό για 15 καρέ
                if frame_counter == FRAMES_TO_CONFIRM:
                    if prediction == "space":
                        current_word += " "
                    elif prediction == "del":
                        current_word = current_word[:-1] # Σβήνει το τελευταίο γράμμα
                    elif prediction != "nothing":
                        current_word += prediction
                    
                    # Δίνουμε μια μικρή παύση για να μην γράψει το ίδιο γράμμα 2η φορά αμέσως
                    frame_counter = -10 
                    
                # Γραφικά: Κουτάκι πάνω αριστερά με τη live πρόβλεψη
                cv2.rectangle(frame_bgr, (10, 10), (180, 80), (0, 0, 0), -1)
                cv2.putText(frame_bgr, f"Live: {prediction}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                
                # Γραφικά: Μπάρα φόρτωσης 
                if frame_counter > 0:
                    bar_width = int((frame_counter / FRAMES_TO_CONFIRM) * 160)
                    cv2.rectangle(frame_bgr, (10, 90), (10 + bar_width, 105), (0, 255, 255), -1)

            # --- ΕΜΦΑΝΙΣΗ ΤΗΣ ΛΕΞΗΣ ΣΤΟ ΚΑΤΩ ΜΕΡΟΣ ---
            h, w = frame_bgr.shape[:2]
            cv2.rectangle(frame_bgr, (0, h - 70), (w, h), (0, 0, 0), -1)
            cv2.putText(frame_bgr, f"Word: {current_word}", (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)

            cv2.imshow("Sign Language Word Builder", frame_bgr)
            
            # --- ΠΛΗΚΤΡΑ ΕΛΕΓΧΟΥ ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):   # τερματισμός
                break
            elif key == ord("c"): # Αν πατήσoυμε το 'c'  η λέξη μηδενίζεται!
                current_word = ""

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()