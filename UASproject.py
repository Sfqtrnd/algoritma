import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Modul 1: Preprocessing Data
# Fungsi untuk menghitung BMI dan menentukan kategori risiko obesitas
def calculate_bmi_and_labels(heights, weights):
    bmi_values = weights / (heights / 100) ** 2
    labels = []
    for bmi in bmi_values:
        if bmi < 18.5:
            labels.append(0)  # Underweight
        elif 18.5 <= bmi < 25:
            labels.append(1)  # Normal weight
        elif 25 <= bmi < 30:
            labels.append(2)  # Overweight
        else:
            labels.append(3)  # Obesity
    return bmi_values, np.array(labels)

# Modul 2: Membagi Data
def split_data(features, labels, test_size=0.3, random_state=42):
    return train_test_split(features, labels, test_size=test_size, random_state=random_state)

# Modul 3: Training Model
def train_svm(X_train, y_train, kernel='rbf', C=1.0):
    svm_model = SVC(kernel=kernel, C=C, random_state=42)
    svm_model.fit(X_train, y_train)
    return svm_model

def train_knn(X_train, y_train, n_neighbors=5):
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(X_train, y_train)
    return knn_model

def train_random_forest(X_train, y_train, n_estimators=100):
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

# Modul 4: Evaluasi Model
def evaluate_model(model, X_test, y_test, class_names):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")

    # Dapatkan label unik dari y_test dan y_pred
    unique_labels = np.unique(np.concatenate((y_test, y_pred)))

    # Filter class_names untuk mencocokkan label unik
    filtered_class_names = [class_names[i] for i in unique_labels]

    print(classification_report(y_test, y_pred, target_names=filtered_class_names))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=filtered_class_names, yticklabels=filtered_class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
    return accuracy

# Modul 5: Visualisasi Akurasi
def plot_accuracies(accuracies, model_names):
    plt.bar(model_names, accuracies, color=['blue', 'orange', 'green'])
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Comparison')
    plt.ylim(0, 1)
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f"{acc:.2f}", ha='center')
    plt.show()

# Main Program
if __name__ == "__main__":

    # Data input (contoh data tinggi badan dalam cm dan berat badan dalam kg)
    heights = np.array([160, 165, 170, 175, 180, 155, 150, 190, 200, 140])
    weights = np.array([50, 60, 70, 80, 90, 40, 45, 100, 110, 35])

    # Step 1: Hitung BMI dan Label
    print("Calculating BMI and labels...")
    bmi_values, labels = calculate_bmi_and_labels(heights, weights)

    # Fitur hanya menggunakan nilai BMI
    features = bmi_values.reshape(-1, 1)

    # Step 2: Split data
    print("Splitting data into training and validation sets...")
    X_train, X_test, y_train, y_test = split_data(features, labels)

    # Step 3: Train models
    print("Training models...")
    svm_model = train_svm(X_train, y_train)
    knn_model = train_knn(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    # Step 4: Evaluate models
    class_names = ['Underweight', 'Normal weight', 'Overweight', 'Obesity']

    print("Evaluating SVM model...")
    svm_accuracy = evaluate_model(svm_model, X_test, y_test, class_names)

    print("Evaluating K-NN model...")
    knn_accuracy = evaluate_model(knn_model, X_test, y_test, class_names)

    print("Evaluating Random Forest model...")
    rf_accuracy = evaluate_model(rf_model, X_test, y_test, class_names)

    # Step 5: Visualize accuracies
    accuracies = [svm_accuracy, knn_accuracy, rf_accuracy]
    model_names = ['SVM', 'K-NN', 'Random Forest']
    plot_accuracies(accuracies, model_names)
