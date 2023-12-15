import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the CSV file containing the time series data
data = pd.read_csv('Data_Random_Food.csv')

# Convert the 'date' column to datetime format
data['Tanggal'] = pd.to_datetime(data['Tanggal'], format='%d/%m/%Y')

# Set the 'date' column as the index
data.set_index('Tanggal', inplace=True)

# Resample the data by day and calculate the sum (or mean, etc.)
daily_data = data.resample('D').sum()  # Change 'sum' to 'mean', etc. as needed

# Reset the index
daily_data.reset_index(inplace=True)

# Extract the time series columns
series = daily_data[['Calories', 'Carbs', 'Prots', 'Fats']].values

# Define window size and label width
if len(series) <= 28:
    label_width = 2
    window_size = int(len(series)/2)

elif len(series) > 28:
    label_width = 7
    window_size = int(len(series)/2)
elif len(series) > 56:
    label_width = 7
    window_size = int(len(series)/3)

input_width = window_size - label_width
shift = 1
def create_lstm_model():
    # Function to create time windows for training data
    def create_time_windows(series=series, window_size=window_size, input_width=input_width, label_width=label_width, shift=shift):
        inputs = []
        labels = []
        for i in range(len(series) - window_size - label_width + 1):
            input_end = i + input_width
            label_start = input_end
            label_end = label_start + label_width
            inputs.append(series[i:input_end])
            labels.append(series[label_start:label_end])
        return np.array(inputs), np.array(labels)

    # Generate data using time windows
    inputs, labels = create_time_windows()

    # Split the data into training, validation, and testing sets (60/20/20)
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(inputs, labels, test_size=0.2, random_state=42)
    train_inputs, val_inputs, train_labels, val_labels = train_test_split(train_inputs, train_labels, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

    # Adjusting the model architecture
    model = tf.keras.models.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True), input_shape=[input_width, 4]),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(label_width * 4, 'sigmoid'),  # Adjust output for 4 features
        tf.keras.layers.Reshape([label_width, 4])
    ])

    # Compile the model
    model.compile(
        loss=tf.keras.losses.MeanAbsoluteError(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )

    # Fit the model using the training data and validate with the validation data
    history = model.fit(
        train_inputs,  # Training input data
        train_labels,  # Training target data
        validation_data=(val_inputs, val_labels),  # Validation data
        epochs=2  # You can adjust the number of epochs
    )

    # Save the model as .h5 file
    model.save('lstm_model.h5')

    return model, test_inputs, test_labels  # Return the trained model, test inputs, test labels, and input data for prediction

# Call the function to create and get the LSTM model along with test data
model, test_inputs, test_labels = create_lstm_model()

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_inputs, test_labels)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Example of preparing input data for prediction (using the last window of the series)
input_data = series[-input_width:].reshape((1, input_width, 4))
predicted_values = model.predict(input_data)
print(predicted_values)
print(predicted_values.shape)

# Create a new figure
plt.figure()

# Plot each column of the predicted values as a separate line
for i in range(predicted_values.shape[2]):
    plt.plot(predicted_values[0, :, i], label=f'Predicted Values {i+1}')

# Add a legend
plt.legend()

# Display the plot
plt.show()