import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load the pretrained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Load an image for prediction (replace 'your_image_path' with actual path)
img_path = 'e:\Solar system\sample.jpg'
img = image.load_img(img_path, target_size=(224, 224))

# Preprocess the image to match model input requirements
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Predict the class of the image
predictions = model.predict(img_array)

# Decode and print the top 3 predicted classes
top_predictions = decode_predictions(predictions, top=3)[0]
print("Predicted Classes:")
for i, (imagenet_id, label, score) in enumerate(top_predictions):
    print(f"{i+1}: {label} ({score*100:.2f}%)")

# Display the input image
plt.imshow(img)
plt.axis('off')
plt.show()
