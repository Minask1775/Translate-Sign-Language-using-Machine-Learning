import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

#  ελληνικά στο τερματικό 
sys.stdout.reconfigure(encoding='utf-8')

# --- Ρυθμίσεις Διαδρομών ---
BASE_DIR = os.path.dirname(__file__)
CSV_FILE = os.path.join(BASE_DIR, "hand_landmarks.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "sign_language_model.pkl")

def main():
    print("1. Φόρτωση δεδομένων από το CSV...")
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print("Σφάλμα: Δεν βρέθηκε το αρχείο hand_landmarks.csv!")
        return

    print("2. Καθαρισμός και Επεξεργασία δεδομένων...")
    # ΦΙΛΤΡΟ: Κρατάμε μόνο τα γράμματα που έχουν 10 ή περισσότερες φωτογραφίες
    label_counts = df['label'].value_counts()
    valid_labels = label_counts[label_counts >= 10].index
    df = df[df['label'].isin(valid_labels)]

    # Χωρισμός σε Features (συντεταγμένες) και Labels (γράμματα)
    X = df.drop('label', axis=1) 
    y = df['label'] 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("3. Εκπαίδευση του Μοντέλου")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("4. Αξιολόγηση Μοντέλου...")
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    
    print("-" * 30)
    print(f"ΑΚΡΙΒΕΙΑ ΜΟΝΤΕΛΟΥ (Accuracy): {score * 100:.2f}%")
    print("-" * 30)

    print("5. Αποθήκευση του εκπαιδευμένου μοντέλου...")
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"ΤΕΛΟΣ! Το μοντέλο αποθηκεύτηκε στο αρχείο: sign_language_model.pkl")

if __name__ == "__main__":
    main()