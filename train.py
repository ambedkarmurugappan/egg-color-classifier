import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Load dataset
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train = train_datagen.flow_from_directory(
    "dataset",
    target_size=(150, 150),
    batch_size=32,
    class_mode="categorical",
    subset='training'
)

val = train_datagen.flow_from_directory(
    "dataset",
    target_size=(150, 150),
    batch_size=32,
    class_mode="categorical",
    subset='validation'
)

# Model (CNN)
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(4, activation='softmax')  # 4 classes
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
history = model.fit(train, validation_data=val, epochs=10)

# Save model
model.save("egg_color_model.h5")

print("Training completed! Model saved as egg_color_model.h5")
print("Class mapping:", train.class_indices)



# Convert model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open("egg_color_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved!")
