
# ANN MODEL FOR CAFFEINE TABLET FORMULATION


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping


# Load dataset

data = pd.read_csv("caffeine_tablet_data.csv")

X = data[['MCC', 'Starch', 'Binder', 'Pressure']].values
y = data[['Hardness', 'Friability', 'Disintegration']].values

# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Normalization: improves ANN performance

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Build ANN model

model = Sequential([
    Dense(32, activation='relu', input_shape=(4,)),  
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(3)     # Outputs: hardness, friability, disintegration
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)


early_stop = EarlyStopping(
    monitor='val_loss',
    patience=30,
    restore_best_weights=True
)

# ------------------------------------------
# Train model

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=500,
    batch_size=8,
    callbacks=[early_stop],
    verbose=1
)


# Evaluate model

loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test MAE: {mae:.4f}")


# Example Prediction

example = np.array([[60, 10, 5, 15]])  # MCC, Starch, Binder, Pressure
example_scaled = scaler.transform(example)

prediction = model.predict(example_scaled)
print("Predicted [Hardness, Friability, Disintegration]:")
print(prediction)
