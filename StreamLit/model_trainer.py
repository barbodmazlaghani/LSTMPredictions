import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense

def create_bilstm_model(input_shape):
    """
    Creates a BiLSTM model.
    """
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        Bidirectional(LSTM(32)),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_and_save_model(data, save_folder="models"):
    """
    Trains the BiLSTM model and saves it.
    
    Args:
    - data: Normalized numpy array with features and labels.
    - save_folder: Directory where the model is saved.
    
    Returns:
    - model_path: Path to the saved model.
    """
    os.makedirs(save_folder, exist_ok=True)
    X_train = data[:, :-1]  # Features
    y_train = data[:, -1]   # Labels

    input_shape = (X_train.shape[1], 1)
    model = create_bilstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    model_path = os.path.join(save_folder, "bilstm_model.h5")
    model.save(model_path)
    return model_path
