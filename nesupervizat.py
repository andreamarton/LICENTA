import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow import keras
from tensorflow.keras.layers import Dropout

def run_model(data, epochs=50):

    #daca sunt valori lipsa se sterg
    data = data.dropna()

    # Separare date și etichete
    y_raw = data.iloc[:, 0].values.astype(int)
    X = data.iloc[:, 1:].values
    y_true = np.where(y_raw == 0, 0, 1)             # 0 = normal, altceva = anomalie

    data_normal = data[data.iloc[:,0] == 0].copy()
    X_normal = data_normal.iloc[:,1:].values

    # Normalizare
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Definire autoencoder
    input_layer = keras.Input(shape=(X.shape[1],))

    # Encoder
    encoded = keras.layers.Dense(64, activation='relu')(input_layer)
    encoded = Dropout(0.05)(encoded)
    encoded = keras.layers.Dense(32, activation='relu')(encoded)
    encoded = Dropout(0.05)(encoded)

    # Decoder
    decoded = keras.layers.Dense(64, activation='relu')(encoded)
    decoded = keras.layers.Dense(X.shape[1], activation='sigmoid')(decoded)

    autoencoder = keras.Model(inputs=input_layer, outputs=decoded)

    # Callback pentru salvarea pierderii pe fiecare epoca
    class LossLogger(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            self.epoch_losses = []
            self.epoch_val_losses = []
        def on_epoch_end(self, epoch, logs=None):
            self.epoch_losses.append(logs["loss"])
            self.epoch_val_losses.append(logs.get("val_loss"))

    loss_logger = LossLogger()
    #compilare model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    autoencoder.compile(optimizer=optimizer, loss='mse')

    # Antrenare autoencoder
    autoencoder.fit(X_scaled, X_scaled, epochs=epochs, batch_size=256,
                    validation_split=0.2, callbacks=[loss_logger])

    # Reconstruire
    X_reconstructed = autoencoder.predict(X_scaled)

    # Calcul eroare MSE per esantion
    mse = np.mean(np.square(X_scaled - X_reconstructed), axis=1).reshape(-1, 1)

    # Antrenare One-Class pe erorile de reconstructie
    ocsvm = OneClassSVM(kernel='rbf', nu=0.1, gamma='auto')
    ocsvm.fit(mse)

    #predictiile
    predictions = ocsvm.predict(mse)
    y_pred = np.where(predictions == 1, 0, 1).astype(int)

    # Calcul metrici
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    accuracy = accuracy_score(y_true, y_pred)

    # Extragem din matricea de confuzie
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

    print("=== REZULTATE MODEL NESUPERVIZAT ===")
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

