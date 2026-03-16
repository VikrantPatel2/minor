# =====================================================
# ANN MODEL FOR CAFFEINE TABLET FORMULATION
# =====================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping


# -----------------------------------------------------
# 1. Load Dataset
# -----------------------------------------------------

data = pd.read_csv("caffeine_tablet_data.csv")

X = data[['MCC', 'Starch', 'Binder', 'Pressure']].values
y = data[['Hardness', 'Friability', 'Disintegration']].values


# -----------------------------------------------------
# 2. Train Test Split
# -----------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -----------------------------------------------------
# 3. Feature Scaling
# -----------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# -----------------------------------------------------
# 4. Build ANN Model
# -----------------------------------------------------

model = Sequential([
    
    Dense(32, activation='relu', input_shape=(4,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    
    Dense(3)   # Outputs: Hardness, Friability, Disintegration
    
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)


# -----------------------------------------------------
# 5. Early Stopping
# -----------------------------------------------------

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=30,
    restore_best_weights=True
)


# -----------------------------------------------------
# 6. Train Model
# -----------------------------------------------------

history = model.fit(
    
    X_train, y_train,
    
    validation_split=0.2,
    
    epochs=500,
    batch_size=8,
    
    callbacks=[early_stop],
    
    verbose=1
    
)


# -----------------------------------------------------
# 7. Evaluate Model
# -----------------------------------------------------

loss, mae = model.evaluate(X_test, y_test, verbose=0)

print("\nModel Evaluation")
print("----------------")
print(f"Test Loss: {loss:.4f}")
print(f"Test MAE: {mae:.4f}")


# -----------------------------------------------------
# 8. Example Prediction
# -----------------------------------------------------

example = np.array([[60, 10, 5, 15]])  # MCC, Starch, Binder, Pressure

example_scaled = scaler.transform(example)

prediction = model.predict(example_scaled)

print("\nPredicted Properties")
print("--------------------")

pred_hardness = prediction[0][0]
pred_friability = prediction[0][1]
pred_disintegration = prediction[0][2]

print("Hardness:", pred_hardness)
print("Friability:", pred_friability)
print("Disintegration:", pred_disintegration)


# -----------------------------------------------------
# 9. Ideal Target Values
# -----------------------------------------------------

IDEAL_RELEASE = 87      # %
IDEAL_HARDNESS = 5.8    # kg/cm2
IDEAL_FRIABILITY = 0.6  # %

print("\nIdeal Targets")
print("-------------")

print("Release (12h):", IDEAL_RELEASE)
print("Hardness:", IDEAL_HARDNESS)
print("Friability:", IDEAL_FRIABILITY)


# -----------------------------------------------------
# 10. Check If Formulation is Ideal
# -----------------------------------------------------

if pred_hardness >= IDEAL_HARDNESS and pred_friability <= IDEAL_FRIABILITY:
    
    print("\nThis formulation is CLOSE TO IDEAL")

else:
    
    print("\nThis formulation needs optimization")


# -----------------------------------------------------
# 11. Plot Training Performance
# -----------------------------------------------------

plt.figure(figsize=(8,5))

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.title("ANN Training Performance")

plt.legend()

plt.show()


# -----------------------------------------------------
# 12. Predicted vs Ideal Comparison
# -----------------------------------------------------

labels = ['Hardness', 'Friability']

predicted = [pred_hardness, pred_friability]
ideal = [IDEAL_HARDNESS, IDEAL_FRIABILITY]

x = np.arange(len(labels))

plt.figure(figsize=(7,5))

plt.bar(x - 0.2, predicted, width=0.4, label='Predicted')
plt.bar(x + 0.2, ideal, width=0.4, label='Ideal')

plt.xticks(x, labels)

plt.ylabel("Value")

plt.title("Predicted vs Ideal Tablet Properties")

plt.legend()

plt.show()


# -----------------------------------------------------
# 13. Formulation Optimization Space
# -----------------------------------------------------

plt.figure(figsize=(7,5))

plt.scatter(
    y[:,0],
    y[:,1],
    label="Training Data"
)

plt.scatter(
    pred_hardness,
    pred_friability,
    color='red',
    s=120,
    label="Predicted Formulation"
)

plt.scatter(
    IDEAL_HARDNESS,
    IDEAL_FRIABILITY,
    color='green',
    s=120,
    label="Ideal Target"
)

plt.xlabel("Hardness")

plt.ylabel("Friability")

plt.title("Formulation Optimization Space")

plt.legend()

plt.show()

# =====================================================
# END OF PROGRAM
# =====================================================
