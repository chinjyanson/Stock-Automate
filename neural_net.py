import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader, TensorDataset
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import math
import csv

class StockPredictor:
    def __init__(self, x, nb_epoch=1000, hidden_layers=[128, 64, 32], learning_rate=0.001, batch_size=64):
        """
        Initialize the model.
        Arguments:
            - x {pd.DataFrame} -- Raw input data used to compute the size of the network.
            - hidden_layers {list} -- Sizes of hidden layers.
            - learning_rate {float} -- Learning rate for the optimizer.
            - batch_size {int} -- Batch size for training.
        """
        self.encoder = LabelBinarizer()

        # Preprocess input and determine input size
        X, _ = self._preprocessor(x, training=True)
        self.input_size = X.shape[2]  # Number of features per time step
        self.output_size = 1
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size

        self.train_losses = []
        self.early_stopping_epoch = None

        # Define LSTM model architecture
        self.hidden_size = hidden_layers[0]  # Use the first hidden layer size for LSTM
        self.num_layers = len(hidden_layers)  # Number of layers for LSTM

        # LSTM layer
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True)

        # Fully connected output layer
        self.fc = nn.Linear(self.hidden_size, self.output_size)

        # Define loss function, optimizer, and scheduler
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.lstm.parameters(), lr=learning_rate, weight_decay=1e-5)  # L2 regularization
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5)

    def _preprocessor(self, x, y=None, training=False):
        """
        Preprocess the input data for the network.
        Arguments:
            - x {pd.DataFrame} -- The input features to preprocess. This DataFrame may contain both numerical
                                and categorical data.
            - y {pd.Series or pd.DataFrame, optional} -- The target labels corresponding to the input data `x`.
                                                        This is required for training but not for inference. Default is None.
            - training {bool} -- Flag indicating whether the function is being used for training (True) or inference (False).
                                If True, the method will compute and store the mean and standard deviation for scaling.
                                If False, it will use the precomputed values for scaling and apply the transformation
                                to categorical features using the stored encoder.
        Returns:
            - x {torch.Tensor} -- The preprocessed input features as a PyTorch tensor, ready to be used in the model.
            - y {torch.Tensor or None} -- The preprocessed target labels as a PyTorch tensor. Returns None if `y` is not provided.
        """
        if torch.is_tensor(x):
            return x, y if torch.is_tensor(y) else torch.tensor(y.values, dtype=torch.float32) 

        x = x.copy()
        for col in x.select_dtypes(include=[np.number]).columns: # Check if column is an integer type
            x[col] = x[col].fillna(x[col].mean())

        x['ocean_proximity'] = x['ocean_proximity'].astype(str)

        # Split numerical and categorical features
        x_numeric = x.select_dtypes(include=[np.number])
        x_categorical = x[['ocean_proximity']]

        if training:
            x_categorical = self.encoder.fit_transform(x_categorical)
            self.mean = x_numeric.mean(axis=0)
            self.std = x_numeric.std(axis=0)
        else:
            unknown_categories = set(x_categorical['ocean_proximity'].unique()) - set(self.encoder.classes_)
            if unknown_categories:
                x_categorical['ocean_proximity'] = x_categorical['ocean_proximity'].apply(lambda x: x if x in self.encoder.classes_ else 'unknown')
            x_categorical = self.encoder.transform(x_categorical)

        x_numeric = (x_numeric - self.mean) / self.std

        x = np.concatenate([x_numeric, x_categorical], axis=1)
        # Reshape for LSTM input: [batch_size, seq_length, input_size]
        sequence_length = 10  # Adjust based on your data and use case
        if x.shape[0] % sequence_length != 0:
            raise ValueError(f"Number of samples ({x.shape[0]}) must be divisible by sequence_length ({sequence_length}).")

        x = x.values.reshape(-1, sequence_length, x.shape[1])  # 3D shape
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y.values, dtype=torch.float32) if isinstance(y, pd.DataFrame) else None
        return x, y

    def forward(self, x):
        # Initialize hidden and cell states for the LSTM
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Pass input through the LSTM layer
        out, _ = self.lstm(x, (h0, c0))  # out: [batch_size, seq_length, hidden_size]

        # Only use the output of the last time step for prediction
        out = self.fc(out[:, -1, :])  # out: [batch_size, output_size]
        return out

    def fit(self, x, y, early_stopping=True, patience=5):
        """
        Train the model on the given data with optional early stopping.

        Arguments:
            - x {pd.DataFrame} -- Input features for training.
            - y {pd.Series or pd.DataFrame} -- Target values for training.
            - early_stopping {bool} -- Whether to use early stopping (default: True).
            - patience {int} -- Number of epochs with no improvement before stopping (default: 5).

        Returns:
            - self {StockPredictor} -- The trained model.
        """
        X, Y = self._preprocessor(x, y=y, training=True)
        dataset = TensorDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.nb_epoch):
            epoch_loss = 0
            for batch_X, batch_Y in dataloader:
                self.optimizer.zero_grad()  # Reset gradients
                predictions = self.model(batch_X)  # Forward pass
                loss = self.loss_fn(predictions, batch_Y)  # Calculate loss
                loss.backward()  # Backward pass
                self.optimizer.step()  # Update weights
                epoch_loss += loss.item()

            epoch_loss /= len(dataloader)
            self.train_losses.append(epoch_loss)
            self.scheduler.step(epoch_loss)  # Adjust learning rate dynamically
            print(f"Epoch {epoch + 1}/{self.nb_epoch}, RMSE Loss: {math.sqrt(epoch_loss):.4f}")

            if early_stopping:
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    self.early_stopping_epoch = epoch
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break
        if self.early_stopping_epoch is None:
            self.early_stopping_epoch = self.nb_epoch
        return self

    def predict(self, x):
        """
        Predict the target values for the given input features.

        Arguments:
            - x {pd.DataFrame} -- Input features for making predictions.

        Returns:
            - predictions {numpy.ndarray} -- Predicted values corresponding to the input features.
        """
        X, _ = self._preprocessor(x, training=False)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X)
        return predictions.numpy()

    def score(self, x, y):
        """
        Evaluate the model's performance on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Input features for evaluation.
            - y {pd.DataFrame} -- True target values corresponding to the input features.

        Returns:
            - rmse {float} -- Root Mean Squared Error (RMSE) between predicted and true values.
            - r2_score {float} -- R-squared score, indicating model fit quality.
        """
        X, Y = self._preprocessor(x, y=y, training=False)
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            predictions = self.model(X)  # Get predictions
        # Calculate RMSE
        rmse = torch.sqrt(self.loss_fn(predictions, Y))
        # Calculate R^2 Score
        ss_res = torch.sum((Y - predictions) ** 2)
        ss_tot = torch.sum((Y - torch.mean(Y)) ** 2)
        r2_score = 1 - ss_res / ss_tot
        return rmse.item(), r2_score.item()

    def plot_loss(self):
        """Plot training loss over epochs."""
        plt.plot(self.train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Function')
        plt.legend()
        plt.show()

    def plot_predictions(self, x, y):
        """
        Plots X against Y with the line of best fit using the model's predictions.

        Arguments:
            x {pd.DataFrame or torch.Tensor} -- Input features for prediction.
            y {pd.DataFrame or torch.Tensor} -- True target values.
        """
        # Ensure the input is preprocessed and predictions are made
        X, Y = self._preprocessor(x, y=y, training=False)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X).numpy()
        # Convert tensors to numpy arrays for plotting if necessary
        if torch.is_tensor(X):
            X = X.numpy()
        if torch.is_tensor(Y):
            Y = Y.numpy()

        # Plot actual vs. predicted values
        plt.figure(figsize=(10, 6))
        plt.scatter(Y, predictions, alpha=0.6, s=10, label="Data Points")
        # Compute the line of best fit using Linear Regression
        lin_reg = LinearRegression()
        lin_reg.fit(Y.reshape(-1, 1), predictions)
        best_fit_line = lin_reg.predict(Y.reshape(-1, 1))

        # Plot the line of best fit
        plt.plot(Y, best_fit_line, color='red', label="Line of Best Fit")

        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title("True vs. Predicted Values with Line of Best Fit")
        plt.legend()
        plt.show()

def save_regressor(trained_model):
    """
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor():
    """
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


def perform_hyperparameter_search(x_train, y_train, k=5, results_file="hyperparameter_results.csv"):
    """
    Performs a hyperparameter search for tuning the regressor and writes results to a CSV file.

    Arguments:
        x_train {pd.DataFrame} -- Input training features.
        y_train {pd.DataFrame} -- Target values for training.
        k {int} -- Number of folds for cross-validation.
        results_file {str} -- Name of the CSV file to save results.

    Returns:
        dict -- Best hyperparameter configuration.
        list -- Results of all configurations.
    """
    # nb_epochs_options = [50]
    # hidden_layers_options = [[64, 32]]
    # learning_rate_options = [0.01]
    # batch_size_options = [64]

    # Example of broader search:
    hidden_layers_options = [[32], [64, 32], [64, 64], [128, 64], [64, 32, 16], [128, 64, 32]]
    learning_rate_options = [0.01, 0.001, 0.0001]
    batch_size_options = [32, 64, 128]

    best_score = float('inf')
    best_params = {}
    # results = []

    # Initialize CSV file and write the header
    with open(results_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["exit_epoch", "hidden_layers", "learning_rate", "batch_size", "avg_score", "avg_r2"])

    # K-Fold Cross-Validation
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)

    for hidden_layers in hidden_layers_options:
        for learning_rate in learning_rate_options:
            for batch_size in batch_size_options:
                fold_scores = []
                fold_r2 = []

                # K-Fold loop
                for train_index, val_index in kfold.split(x_train):
                    x_train_fold, x_val_fold = x_train.iloc[train_index], x_train.iloc[val_index]
                    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

                    # Initialize and train the regressor
                    regressor = StockPredictor(x_train_fold, hidden_layers=hidden_layers,
                                            learning_rate=learning_rate, batch_size=batch_size)
                    regressor.fit(x_train_fold, y_train_fold)
                    score, r2 = regressor.score(x_val_fold, y_val_fold)
                    fold_scores.append(score)
                    fold_r2.append(r2)

                # Calculate average score across folds
                avg_score = sum(fold_scores) / k
                avg_r2 = sum(fold_r2) / k
                # results.append(( ))

                # Write results to CSV
                with open(results_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([regressor.early_stopping_epoch+1, hidden_layers, learning_rate, batch_size, avg_score, avg_r2])

                # Update best score and parameters if the current avg_score is lower
                if avg_score < best_score:
                    best_score = avg_score
                    best_params = {
                        "hidden_layers": hidden_layers,
                        "learning_rate": learning_rate,
                        "batch_size": batch_size
                    }
                    print(f"Tested config: {best_params}, Average Score={avg_score}")

    print(f"Best params: {best_params} with Average Score={best_score}")
    return best_params

def train_and_evaluate_regressor(hidden_layers=[128, 64, 32], learning_rate=0.01, batch_size=128, patience=5, test_size=0.2):
    """
    Train and evaluate a regressor model with the given hyperparameters.
    
    Arguments:
        - hidden_layers {list} -- List of hidden layer sizes in the neural network.
        - learning_rate {float} -- Learning rate for the optimizer.
        - batch_size {int} -- Number of samples per batch during training.
        - patience {int} -- Number of epochs to wait for improvement before early stopping.
        - test_size {float} -- Fraction of the data to use for testing (default 0.2).
    """
    output_label = "median_house_value"
    data = pd.read_csv("housing.csv")

    # Separate features and target
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

    # Initialize and train the regressor
    regressor = StockPredictor(
        x_train,
        hidden_layers=hidden_layers,
        learning_rate=learning_rate,
        batch_size=batch_size
    )
    regressor.fit(x_train, y_train, patience=patience)

    # Save the trained model
    save_regressor(regressor)

    # Evaluate the model on the training and testing sets
    train_score, train_r2 = regressor.score(x_train, y_train)
    test_score, test_r2 = regressor.score(x_test, y_test)

    # Print performance metrics
    print(f"Train RMSE: {train_score:.4f}, Train R²: {train_r2:.4f}")
    print(f"Test RMSE: {test_score:.4f}, Test R²: {test_r2:.4f}")

    # Plot loss and predictions
    regressor.plot_loss()
    regressor.plot_predictions(x, y)


def example_main():
    output_label = "median_house_value"
    data = pd.read_csv("housing.csv")

    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    best_params = perform_hyperparameter_search(x_train, y_train, k=5)

    # Train the final model with the best parameters
    regressor = StockPredictor(
        x_train,
        hidden_layers=best_params["hidden_layers"],
        learning_rate=best_params["learning_rate"],
        batch_size=best_params["batch_size"]
    )
    regressor.fit(x_train, y_train, patience=5)
    save_regressor(regressor)

    # Evaluate on the test set
    train_score, train_r2 = regressor.score(x_train, y_train)
    test_score, test_r2 = regressor.score(x_test, y_test)
    print(f"Train RMSE: {train_score:.4f}, Train R²: {train_r2:.4f}")
    print(f"Test RMSE: {test_score:.4f}, Test R²: {test_r2:.4f}")

    regressor.plot_loss()
    regressor.plot_predictions(x, y)

if __name__ == "__main__":
    train_and_evaluate_regressor(hidden_layers=[128, 64, 32], learning_rate=0.01, batch_size=128, patience=5, test_size=0.2)
    # example_main()

