import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# 1. Load the Dataset
try:
    df = pd.read_csv('/Users/vinayakprakash/Documents/files/fertilizername/model/Fertilizer Prediction.csv')
except FileNotFoundError:
    print("Error: 'Fertilizer Prediction.csv' not found. Please download it from Kaggle.")
    exit()

# 2. CLEAN COLUMN NAMES (Fixing Dataset Errors)
# The Kaggle dataset has 'Temparature' (typo) and 'Humidity ' (space at end)
df.columns = df.columns.str.strip() # Removes spaces like 'Humidity ' -> 'Humidity'
df.rename(columns={'Temparature': 'Temperature'}, inplace=True) # Fix spelling

# 3. DROP TEMPERATURE
# We keep Humidity, Moisture, etc., but drop Temperature as you requested.
if 'Temperature' in df.columns:
    df = df.drop(columns=['Temperature'])

print("Columns used for training:", list(df.columns))

# 4. ENCODE CATEGORICAL DATA
# Convert text (Sandy, Maize) into numbers (0, 1)
le_soil = LabelEncoder()
df['Soil Type'] = le_soil.fit_transform(df['Soil Type'])

le_crop = LabelEncoder()
df['Crop Type'] = le_crop.fit_transform(df['Crop Type'])

le_ferti = LabelEncoder()
df['Fertilizer Name'] = le_ferti.fit_transform(df['Fertilizer Name'])

# 5. PREPARE INPUTS AND TARGET
# Inputs: Humidity, Moisture, Soil Type, Crop Type, Nitrogen, Potassium, Phosphorous
X = df[['Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']]
y = df['Fertilizer Name']

# 6. TRAIN MODEL (XGBoost)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Model...")
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# 7. SAVE EVERYTHING
joblib.dump(model, 'ferti_model.pkl')
joblib.dump(le_soil, 'enc_soil.pkl')
joblib.dump(le_crop, 'enc_crop.pkl')
joblib.dump(le_ferti, 'enc_ferti.pkl')

print(f"Success! Model Accuracy: {accuracy_score(y_test, model.predict(X_test))*100:.2f}%")