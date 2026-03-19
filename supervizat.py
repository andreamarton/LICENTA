import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback


def run_model(data, epochs=50):
    # impart datele in coloana unde apar sau nu anomaliile si celelate caracteristici
    y = data.iloc[:, 0].values
    X = data.iloc[:, 1:].values


    # Normalizare date
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)


    # transformare etichete in valori numerice
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    # impartire in seturi de antrenare și test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_categorical, test_size=0.2, random_state=42
    )

    # Callback pentru salvarea pierderii in fiecare epoca
    class LossLogger(Callback):
        def on_train_begin(self, logs=None):
            self.epoch_losses = []
            self.epoch_val_losses = []
        def on_epoch_end(self, epoch, logs=None):
            self.epoch_losses.append(logs["loss"])
            self.epoch_val_losses.append(logs.get("val_loss"))

    loss_logger = LossLogger()

    # Definire model Keras
    model = Sequential([
        Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(16, activation='relu'),
        Dense(y_train.shape[1], activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Antrenare model
    model.fit(X_train, y_train, epochs=epochs, batch_size=256,
              validation_data=(X_test, y_test), callbacks=[loss_logger])

    # Predictii
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    #convertire calse in binar
    y_true = np.where(y_true == 0, 0, 1)
    y_pred = np.where(y_pred == 0, 0, 1)


    # Evaluare
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    accuracy = accuracy_score(y_true, y_pred)

    # Extragem metrici din matricea de confuzie
    if conf_matrix.shape==(2,2):
        TN,FP,FN,TP=conf_matrix.ravel()
    else:
        TN=FP=FN=TP=0

    if (TP + FN) > 0:
        recall = TP / (TP + FN)
    else:
        recall = 0

    if (TP + FP) > 0:
        precision = TP / (TP + FP)
    else:
        precision = 0

    if (precision + recall) > 0:
        f1_score = 2 * precision * recall / (precision + recall)
    else:
        f1_score = 0

    if (FP + TN) > 0:
        fpr = FP / (FP + TN)
    else:
        fpr = 0

    tpr = recall

    if (TN + FP) > 0:
        tnr = TN / (TN + FP)
    else:
        tnr = 0

    if (FN + TP) > 0:
        fnr = FN / (FN + TP)
    else:
        fnr = 0


    print("=== REZULTATE MODEL SUPERVIZAT ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall (TPR): {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

    return (
    accuracy,
    conf_matrix,
    pd.Series(loss_logger.epoch_losses, name='loss'),
    pd.DataFrame(X_scaled),
    pd.DataFrame({"Real": y_true, "Predict": y_pred}),
    recall,
    precision,
    f1_score,
    fpr,
    tpr,
    tnr,
    fnr,
    pd.Series(loss_logger.epoch_val_losses, name='val_loss')
)

