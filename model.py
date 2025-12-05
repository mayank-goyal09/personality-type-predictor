import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

# ============================================
# PERSONALITY TYPE PREDICTOR - MODEL TRAINING
# ============================================

print("🧠 Personality Type Predictor - Model Training")
print("=" * 50)

# ========== 1. LOAD DATA ==========
print("\n📂 Step 1: Loading dataset...")

# TODO: Replace with your actual dataset path
# Expected columns: ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism', 'personality_type']

try:
    df = pd.read_csv('data/personality_data.csv')
    print(f"✅ Dataset loaded successfully! Shape: {df.shape}")
except FileNotFoundError:
    print("⚠️ Dataset not found. Creating sample data for demonstration...")
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    df = pd.DataFrame({
        'openness': np.random.randint(0, 11, n_samples),
        'conscientiousness': np.random.randint(0, 11, n_samples),
        'extraversion': np.random.randint(0, 11, n_samples),
        'agreeableness': np.random.randint(0, 11, n_samples),
        'neuroticism': np.random.randint(0, 11, n_samples)
    })
    
    # Create synthetic personality types based on traits
    def assign_personality(row):
        if row['openness'] > 7 and row['conscientiousness'] > 7:
            return 'Analyst'
        elif row['agreeableness'] > 7 and row['extraversion'] > 7:
            return 'Diplomat'
        elif row['conscientiousness'] > 7 and row['neuroticism'] < 4:
            return 'Sentinel'
        elif row['extraversion'] > 7 and row['openness'] > 6:
            return 'Explorer'
        else:
            return 'Other'
    
    df['personality_type'] = df.apply(assign_personality, axis=1)
    
    # Save sample data
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/personality_data.csv', index=False)
    print(f"✅ Sample dataset created and saved! Shape: {df.shape}")

# ========== 2. DATA EXPLORATION ==========
print("\n🔍 Step 2: Data Exploration")
print(f"Dataset Info:")
print(df.info())
print(f"\nPersonality Type Distribution:")
print(df['personality_type'].value_counts())
print(f"\nMissing Values:")
print(df.isnull().sum())

# ========== 3. DATA PREPROCESSING ==========
print("\n🧹 Step 3: Data Preprocessing")

# Handle missing values (if any)
df = df.dropna()

# Separate features and target
X = df[['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']]
y = df['personality_type']

print(f"✅ Features shape: {X.shape}")
print(f"✅ Target shape: {y.shape}")

# Encode target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"\nEncoded personality types: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("✅ Features scaled using StandardScaler")

# ========== 4. TRAIN-TEST SPLIT ==========
print("\n✂️ Step 4: Splitting data into train and test sets (80-20)")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"✅ Training set size: {X_train.shape[0]} samples")
print(f"✅ Testing set size: {X_test.shape[0]} samples")

# ========== 5. MODEL TRAINING ==========
print("\n🤖 Step 5: Training Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    multi_class='multinomial',
    solver='lbfgs'
)

model.fit(X_train, y_train)
print("✅ Model training completed!")

# ========== 6. MODEL EVALUATION ==========
print("\n📊 Step 6: Model Evaluation")

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Accuracy
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"\n🎯 Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"🎯 Testing Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Classification Report
print("\n📋 Classification Report (Test Set):")
print(classification_report(y_test, y_pred_test, target_names=label_encoder.classes_))

# Confusion Matrix
print("\n🔢 Confusion Matrix (Test Set):")
print(confusion_matrix(y_test, y_pred_test))

# ========== 7. SAVE MODEL ==========
print("\n💾 Step 7: Saving model and preprocessors...")

os.makedirs('models', exist_ok=True)

# Save model
with open('models/personality_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✅ Model saved to 'models/personality_model.pkl'")

# Save scaler
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Scaler saved to 'models/scaler.pkl'")

# Save label encoder
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print("✅ Label encoder saved to 'models/label_encoder.pkl'")

print("\n" + "=" * 50)
print("🎉 Model training completed successfully!")
print("📱 You can now run the Streamlit app: streamlit run app.py")
print("=" * 50)