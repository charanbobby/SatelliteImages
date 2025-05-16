# Satellite Imagery Land Use Classification

This project implements deep learning models to classify land uses and land covers from satellite imagery. It includes both a custom CNN model and a transfer learning approach using MobileNetV2.

https://youtu.be/ghLbLN6X2mE

## Project Overview

The application classifies satellite imagery into 21 different land use categories:
- Agricultural
- Airplane
- Baseball Diamond
- Beach
- Buildings
- Chaparral
- Dense Residential
- Forest
- Freeway
- Golf Course
- Harbor
- Intersection
- Medium Residential
- Mobile Home Park
- Overpass
- Parking Lot
- River
- Runway
- Sparse Residential
- Storage Tanks
- Tennis Court

## Features

- Data preprocessing and exploration
- Custom CNN architecture for image classification
- Transfer learning with MobileNetV2 pre-trained model
- Model training with TensorBoard logging and early stopping
- Real-time classification using screen capture
- Interactive visualization with OpenCV
- Top-5 prediction display

## Installation

1. Clone this repository
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

Alternatively, you can install dependencies individually:

```bash
pip install tensorflow numpy matplotlib scikit-learn mss opencv-python tensorflow-hub pillow
```

## Dataset Structure

The dataset should be organized in the following structure:
```
data/
└── Satellite Image Data/
    └── archive/
        └── images/
            ├── agricultural/
            │   ├── agricultural_000001.png
            │   ├── agricultural_000002.png
            │   └── ...
            ├── airplane/
            └── ...
```

Each class folder contains 500 satellite images in PNG format (128x128 pixels).

## Usage

### Data Preparation

The script converts images to numpy arrays and saves them as .npy files for faster processing:

```python
# Create the npy files for all images and labels
for label in LABEL_MAP:
    for land in range(500):
        path = os.path.join(IMAGE_PATH, '{}/{}_000{:03d}.png'.format(label, label, land+1))
        y.append(LABEL_MAP[label])
        img = conv_to_array(path)
        npy_path = os.path.join(IMAGE_PATH, '{}/{}_000{:03d}'.format(label, label, land+1))
        np.save(npy_path, img)
```

### Training a Custom CNN Model

```python
# Build and train the custom CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=3, input_shape=(128,128,3), activation='relu', padding='same'))
model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)))
model.add(Dropout(0.5))
model.add(Dense(21, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=1e-2), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, callbacks=[tf_callback, early_stopping], 
          epochs=200, batch_size=64, validation_split=0.10)
```

### Transfer Learning with MobileNetV2

```python
# Use pre-trained MobileNetV2 for transfer learning
feature_extractor_layer = KerasLayer(feature_extractor_model,
                                    input_shape=(128,128,3),
                                    trainable=False)
model = Sequential()
model.add(feature_extractor_layer)
model.add(Dense(21, activation='softmax'))

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

model.fit(X_train, y_train, validation_split=0.1, epochs=12, batch_size=64)
```

### Real-time Classification

The project includes functionality to capture a portion of the screen and classify the land use in real-time:

```python
# Capture screen and get predictions
def get_gmap_pred():
    with mss.mss() as sct:
        # Configure the screen capture area
        monitor = {
            "top": mon["top"] + 300,
            "left": mon["left"] + 100,
            "width": 512,
            "height": 512,
            "mon": monitor_number,
        }
        img = sct.grab(monitor)
        output = os.path.join("images/sct-mon{}.png".format(monitor_number))
        mss.tools.to_png(img.rgb, img.size, output=output)
        top5_name, top5_percent = get_top_5_preds('images/sct-mon3.png')
        return top5_name, top5_percent
```

### Interactive Visualization

The application provides an interactive OpenCV window to visualize predictions:

```python
# Press 'c' to capture and classify, 'q' to quit
while True:
    key = cv2.waitKey(0) & 0xFF 
    if key == ord('c'):
        # Capture the current image and send to model for prediction
        IMG_PATH = os.path.join('images/sct-mon3.png')
        IMG = cv2.imread(IMG_PATH)
        cv2.imshow('CV2 Windows', IMG)
        cv2.destroyAllWindows()
        draw_rectangle_preds(IMG_PATH)
    elif key == ord('q'):
        break
```

## Model Performance

The repository includes two trained models:
1. `sequential_33.keras` - Custom CNN model 
2. `XferSequential.keras` - Transfer learning model with MobileNetV2

The transfer learning approach generally achieves higher accuracy with fewer epochs of training.

## Directory Structure

```
SatelliteImages/
├── data/                      # Dataset directory
│   └── Satellite Image Data/
│       └── archive/
│           ├── images/        # Original images organized by class
│           ├── images_train_test_val/ # Split dataset
│           ├── label_map.json # Class mapping
│           ├── readme.txt     # Dataset information
│           ├── test.csv       # Test set metadata
│           ├── train.csv      # Training set metadata
│           └── validation.csv # Validation set metadata
├── logs/                      # TensorBoard logs
├── models/                    # Saved models
├── .gitignore                 # Git ignore file
├── SatelliteImageClassification.ipynb # Main Jupyter notebook
├── SatelliteImagesScreenRecording.mp4 # Demo video
├── SatelliteImagesWorkspace.code-workspace # VS Code workspace
├── readme-md.md               # Project documentation
└── requirements.txt           # Dependencies
```

## Demo

A video demonstration of the application in action is included in the project:

```
SatelliteImagesScreenRecording.mp4
```

This recording shows the real-time classification system working with different satellite images, demonstrating how the model identifies various land use categories from satellite imagery and displays confidence scores.

To view the demo:
1. Clone the repository
2. Open the video file at `SatelliteImagesScreenRecording.mp4`
