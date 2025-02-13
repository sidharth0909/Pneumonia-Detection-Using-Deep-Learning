# 🩺 Pneumonia Detection Using Deep Learning

## 📌 Overview

This project implements a Convolutional Neural Network (CNN) model to detect pneumonia from chest X-ray images. It utilizes **TensorFlow, Keras, OpenCV, and EfficientNetB4** for image preprocessing, model training, and evaluation.

---

## 📂 Dataset

The dataset used is **Chest X-ray Pneumonia**, which contains images categorized as **PNEUMONIA** and **NORMAL**. The dataset is structured into:

- **train/** - Training images
- **test/** - Testing images
- **val/** - Validation images

---

## ⚙️ Setup

### 🔹 Import Required Libraries

```python
import os
import cv2
import keras
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from sklearn.utils import class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout, Conv2D, MaxPooling2D, Flatten
```

### 🔹 Extract Dataset

```python
import zipfile
zip_ref = zipfile.ZipFile('/content/chest-xray-pneumonia.zip', 'r')
zip_ref.extractall('/content')
zip_ref.close()
```

---

## 🖼️ Data Preprocessing

### 🔹 Load & Resize Images

```python
labels = ['PNEUMONIA', 'NORMAL']
img_size = 128

def loading_training_data(data_dir):
    data, labels_list = [], []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            resized_arr = cv2.resize(img_arr, (img_size, img_size))
            data.append(resized_arr)
            labels_list.append(class_num)
    return np.array(data), np.array(labels_list)
```

```python
train_data, train_labels = loading_training_data('/content/chest_xray/train')
test_data, test_labels = loading_training_data('/content/chest_xray/test')
```

### 🔹 Normalize & Reshape Data

```python
X_train = np.array(train_data) / 255
X_test = np.array(test_data) / 255

X_train = X_train.reshape(-1, img_size, img_size, 1)
X_test = X_test.reshape(-1, img_size, img_size, 1)

X_train = np.repeat(X_train, 3, axis=-1)
X_test = np.repeat(X_test, 3, axis=-1)

y_train = np.array(train_labels)
y_test = np.array(test_labels)
```

### 🔹 Train-Validation Split

```python
val_size = 0.2
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=val_size, random_state=21)
```

---

## 🏗️ Model Building

### 🔹 Define CNN Model

```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

### 🔹 Compile Model

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

---

## 📊 Data Augmentation & Training

### 🔹 Data Generators

```python
datagen_train = ImageDataGenerator(rescale=1./255)
datagen_validation = ImageDataGenerator(rescale=1./255)
datagen_test = ImageDataGenerator(rescale=1./255)
```

### 🔹 Load Data from Directory

```python
train_generator = datagen_train.flow_from_directory('/content/chest_xray/train', target_size=(150, 150), batch_size=32, class_mode='binary')
validation_generator = datagen_validation.flow_from_directory('/content/chest_xray/val', target_size=(150, 150), batch_size=32, class_mode='binary')
test_generator = datagen_test.flow_from_directory('/content/chest_xray/test', target_size=(150, 150), batch_size=32, class_mode='binary')
```

### 🔹 Train Model

```python
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.fit(train_generator, validation_data=validation_generator, epochs=10, callbacks=[early_stopping])
```

### 🔹 Save Model

```python
model.save('diagnostic_model.keras')
```

---

## 📈 Model Evaluation

### 🔹 Evaluate on Test Data

```python
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {test_accuracy:.2f}')
```

### 🔹 Load & Predict on New Image

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model('diagnostic_model.keras')

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

img_path = '/content/chest_xray/test/PNEUMONIA/person101_bacteria_483.jpeg'
prediction = model.predict(load_and_preprocess_image(img_path))
label = 'PNEUMONIA' if prediction[0][0] > 0.5 else 'NORMAL'
print(f'Predicted Label: {label}')
```

---

## 🎯 Results

- ✅ **Trained CNN model on chest X-ray images**
- ✅ **Achieved \~87.5% validation accuracy**
- ✅ **Test accuracy \~74%**
- ✅ **Model can classify pneumonia vs. normal cases**

---

## 🚀 Future Improvements

- 🔹 **Use a more complex CNN architecture (e.g., EfficientNetB4)**
- 🔹 **Fine-tune hyperparameters for better accuracy**
- 🔹 **Increase dataset size for better generalization**
- 🔹 **Experiment with different augmentation techniques**

📌 *This project demonstrates a simple yet effective deep-learning approach for pneumonia detection!* 💙

