import os
import sys
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Διορθώνει τα ελληνικά στο τερματικό
sys.stdout.reconfigure(encoding='utf-8')

# --- Διαδρομές ---
BASE_DIR = os.path.dirname(__file__)
CSV_FILE = os.path.join(BASE_DIR, "hand_landmarks.csv")
MODEL_PATH = os.path.join(BASE_DIR, "sign_language_model.pkl")

def main():
    print("1. Φόρτωση δεδομένων και του Μοντέλου...")
    df = pd.read_csv(CSV_FILE)
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    # Ίδιος καθαρισμός με την εκπαίδευση
    label_counts = df['label'].value_counts()
    valid_labels = label_counts[label_counts >= 10].index
    df = df[df['label'].isin(valid_labels)]

    X = df.drop('label', axis=1)
    y = df['label']

    # Χωρίζουμε με τον ΙΔΙΟ ακριβώς τρόπο (random_state=42) για να βρούμε το ίδιο "διαγώνισμα" (test set)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("2. Το μοντέλο λύνει το διαγώνισμα (Test Set)...")
    y_pred = model.predict(X_test)

    # --- ΔΗΜΙΟΥΡΓΙΑ ΠΙΝΑΚΑ ΣΥΓΧΥΣΗΣ (CONFUSION MATRIX) ---
    print("3. Δημιουργία Γραφημάτων...")
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    
    plt.figure(figsize=(14, 10))
    # Φτιάχνουμε τον χρωματιστό πίνακα με το seaborn
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=model.classes_, 
                yticklabels=model.classes_)
    
    plt.title('Πίνακας Σύγχυσης (Confusion Matrix) - Νοηματική Γλώσσα', fontsize=16)
    plt.xlabel('Τι Μάντεψε το Μοντέλο (Predicted Label)', fontsize=12)
    plt.ylabel('Τι Ήταν Στην Πραγματικότητα (True Label)', fontsize=12)
    plt.tight_layout()
    
    # Αποθήκευση της εικόνας!
    plot_path = os.path.join(BASE_DIR, "confusion_matrix.png")
    plt.savefig(plot_path, dpi=300)
    print(f"-> Η εικόνα αποθηκεύτηκε στο: {plot_path}")

    # --- ΕΚΤΥΠΩΣΗ ΑΝΑΛΥΤΙΚΗΣ ΑΝΑΦΟΡΑΣ ---
    print("\n" + "="*50)
    print("ΑΝΑΛΥΤΙΚΗ ΑΝΑΦΟΡΑ ΑΚΡΙΒΕΙΑΣ ΑΝΑ ΓΡΑΜΜΑ")
    print("="*50)
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()