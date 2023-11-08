# iris_classification.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow import keras
from flask import Flask, render_template, request

# Load the Iris dataset
data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                   header=None, names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])

# Split the data into features and labels
X = data.drop("species", axis=1)
y = data["species"]

# Encode target labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the neural network
model = keras.Sequential([
    keras.layers.Input(shape=(4,)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100)

# Evaluate the model
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {test_accuracy}")

# Save the model for later use
model.save('model.h5')

app = Flask(__name__)
model = keras.models.load_model('model.h5')

@app.route('/')
def index():
    return render_template('index.html', prediction="")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(input_data)
        species = np.argmax(prediction)

        species_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']
        predicted_species = species_names[species]

        return render_template('index.html', prediction=f"Predicted species: {predicted_species}")
    else:
        return "Invalid request"

if __name__ == '__main__':
    app.run(debug=True)
