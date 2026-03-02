import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import os

# Define file paths
DATA_FILE = "sign_data.csv"
MODEL_DIR = "model"
MODEL_FILE = os.path.join(MODEL_DIR, "sign_model.pkl")

def train():
    # Ensure the model directory exists
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    print("Loading dataset...")
    try:
        # Load the CSV data using pandas
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find '{DATA_FILE}'. Please run collect_data.py first to gather training data!")
        return

    # Basic data validation
    if len(df) == 0:
        print("Dataset is empty. Please collect more data.")
        return

    # Count samples per label group
    counts = df['label'].value_counts()
    print(f"\nDataset loaded. Found {len(df)} total samples.")
    print("Samples per label:")
    print(counts)
    print("\nPreparing to train model...")

    if len(counts) < 2:
        print("Note: You need at least 2 distinct labels to train an accurate classifier effectively! Keep collecting data!")

    # Check for underlying class imbalance
    for label, count in counts.items():
        if count < 200:
            print(f"WARNING: Class '{label}' has only {count} samples. Consider collecting >= 200 samples for better accuracy.")

    # Separate features (X) from the label (y)
    # y is our target class (e.g. 'A'), X is the 63 flattened coordinates (features)
    y = df['label']
    X = df.drop('label', axis=1)

    # Split data into training and testing sets
    # 80% used for the network to 'learn', 20% to evaluate if it actually understands unseen data
    # Stratify makes sure we have a balanced representation in both test and train sets
    stratify_target = y if (len(counts)>1 and min(counts)>1) else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_target)

    # Initialize the MLPClassifier
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42
    )

    print("Training MLP classifier... ", end="")
    # Train the model with the learning dataset
    model.fit(X_train, y_train)
    print("Done!")

    # Ask the trained model to predict signs for the testing set (X_test) we held back
    predictions = model.predict(X_test)

    # Calculate exactly how accurate it was at guessing those signs (y_test)
    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    print("\n------------------------------")
    print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")
    print("------------------------------")
    print("Confusion Matrix:")
    print(cm)
    print("------------------------------\n")

    # Serialize (save) the trained model to a file so a real web/backend server can load it later!
    print(f"Saving finalized model to => {MODEL_FILE}")
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
        
    print("\nAll done! Your custom sign language AI is successfully built.")

if __name__ == "__main__":
    train()
