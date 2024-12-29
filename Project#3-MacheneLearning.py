import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

def load_images_and_labels(folder, target_size=(128, 128)):
    images = []
    labels = []
    emotions = ['neutral', 'happy', 'sad', 'angry']
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.pgm'):
                emotion = next((emo for emo in emotions if emo in file), None)
                if emotion:
                    img_path = os.path.join(root, file)
                    img = load_img(img_path, color_mode='grayscale', target_size=target_size)
                    img_array = img_to_array(img)
                    images.append(img_array)
                    labels.append(emotion)
    return np.array(images), np.array(labels)

# Directory containing the dataset
data_dir = r"C:\Users\Zakaria\Desktop\faces-#3\faces"

# Load images and labels
images, labels = load_images_and_labels(data_dir)

# Normalize the images
images = images / 255.0

# Encode labels to integers and then to categorical data
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
labels = to_categorical(labels)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Build the neural network
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(4, activation='softmax')  # Four output neurons for four classes
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32
)

# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Define the path to a test image
test_image_path = r"C:\Users\Zakaria\Desktop\sm3.pgm" # Change this to the actual image path

# Load and preprocess the test image
test_image = load_img(test_image_path, color_mode='grayscale', target_size=(128, 128))
test_image_array = img_to_array(test_image) / 255.0  # Normalize the image
test_image_array = np.expand_dims(test_image_array, axis=0)  # Expand dimensions to match the model's input shape

# Display the test image
plt.imshow(img_to_array(test_image), cmap='gray')
plt.title('Actual Emotion: happy')  # Replace 'Angry' with the actual emotion of the image
plt.axis('off')
plt.show()

# Predict the emotion using the model
predicted_probabilities = model.predict(test_image_array)
predicted_class = np.argmax(predicted_probabilities)

# Get the label of the predicted class
predicted_label = label_encoder.inverse_transform([predicted_class])[0]

# Display the prediction
print(f"Predicted Emotion: {predicted_label}")
