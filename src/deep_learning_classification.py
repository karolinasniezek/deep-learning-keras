import keras
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import shap

keras.utils.set_random_seed(43)
tf.config.experimental.enable_op_determinism()

dataset = pd.read_csv("../data/cancer.csv")

X = dataset.drop(["diagnosis", "id"], axis=1)
y = dataset["diagnosis"]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation="relu", input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer= tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="binary_crossentropy",
    metrics= ["accuracy"]
)

history = model.fit(
X_train, y_train, batch_size=32, epochs=30, validation_split=0.1, verbose=1)

preds = model.predict(X_test).reshape(-1)
preds_binary = (preds > 0.5).astype(int)

cr = classification_report(y_test, preds_binary, target_names=label_encoder.classes_)
cm = confusion_matrix(y_test, preds_binary)

print(cr)
print(cm)

# =====================
# CONFUSION MATRIX PLOT
# =====================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Benign\n(Low Risk)",
        "Malignant\n(High Risk)"
    ]
)

fig, ax = plt.subplots(figsize=(6, 5))

disp.plot(
    cmap="Greens",
    ax=ax,
    colorbar=False
)


plt.tight_layout()

plt.savefig(
    "../figures/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label = "Train accuracy")
plt.plot(history.history['val_accuracy'], label = "Val accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Model training accuracy")
plt.savefig('../figures/model-training-accuracy.png')

# =====================
# SHAP
# =====================

explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test[:100])
shap.summary_plot(shap_values, features=X_test[:100], feature_names=X.columns, show=False)

plt.tight_layout()

plt.savefig(
    "../figures/shap_summary_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()