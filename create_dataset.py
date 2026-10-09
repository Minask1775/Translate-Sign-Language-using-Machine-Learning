import os
import csv
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Ρυθμίσεις Διαδρομών ---
DATASET_DIR = "dataset" # Ο φάκελος που μόλις έφτιαξες με τις εικόνες
CSV_FILE = "hand_landmarks.csv" # Το αρχείο που θα παραχθεί
MODEL_PATH = "hand_landmarker.task"

def main():
    # 1. Ρύθμιση του MediaPipe για ΣΤΑΤΙΚΕΣ ΕΙΚΟΝΕΣ (IMAGE mode)
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1, # Ψάχνουμε 1 χέρι ανά φωτογραφία
        min_hand_detection_confidence=0.5
    )

    print("Ξεκινάει η δημιουργία του Dataset... Αυτό θα πάρει λίγη ώρα!\n")

    # 2. Άνοιγμα του αρχείου CSV για εγγραφή
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        
        # Φτιάχνουμε τις επικεφαλίδες: label, x0, y0, z0, x1, y1, z1...
        header = ['label']
        for i in range(21):
            header.extend([f'x{i}', f'y{i}', f'z{i}'])
        writer.writerow(header)

        # 3. Ενεργοποίηση του MediaPipe
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            
            # Ψάχνουμε όλους τους υποφακέλους (A, B, C...)
            for label in sorted(os.listdir(DATASET_DIR)):
                label_dir = os.path.join(DATASET_DIR, label)
                
                # Αν δεν είναι φάκελος  το αγνοούμε
                if not os.path.isdir(label_dir): 
                    continue
                
                print(f"Επεξεργασία γράμματος: {label} ... ", end="")
                
                success_count = 0 # Μετρητής για το πόσες εικόνες διαβάστηκαν σωστά
                
                # 4. Διαβάζουμε κάθε εικόνα μέσα στον φάκελο του γράμματος
                for image_name in os.listdir(label_dir):
                    image_path = os.path.join(label_dir, image_name)
                    img = cv2.imread(image_path)
                    
                    if img is None: 
                        continue

                    # Μετατροπή BGR (OpenCV) σε RGB (MediaPipe)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                    
                    # 5. Εντοπισμός των 21 σημείων
                    result = landmarker.detect(mp_image)

                    # Αν το MediaPipe βρήκε όντως χέρι στην εικόνα
                    if result.hand_landmarks:
                        hand_lms = result.hand_landmarks[0]
                        row = [label] # Ξεκινάμε τη γραμμή στο Excel με το όνομα του γράμματος (π.χ. 'A')
                        
                        # Βάζουμε τις συντεταγμένες x, y, z για κάθε ένα από τα 21 σημεία
                        for lm in hand_lms:
                            row.extend([lm.x, lm.y, lm.z]) 
                            
                        writer.writerow(row)
                        success_count += 1
                
                print(f"Βρέθηκαν χέρια σε {success_count} εικόνες.")

    print(f"\nΤΕΛΟΣ! Το αρχείο {CSV_FILE} δημιουργήθηκε επιτυχώς!")

if __name__ == "__main__":
    main()