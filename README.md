# Handwritten Digit Recognition using Deep Learning

**Author:** Aayush Kumar  
**Course level:** B.Tech First Year  
**Domain:** Artificial Intelligence / Deep Learning

## Project Overview

This project uses a **Convolutional Neural Network (CNN)** to recognize handwritten digits from 0 to 9. The model is trained on the MNIST dataset and can be used through a simple Streamlit web application.

## Features

- Trains a CNN on the MNIST handwritten-digit dataset.
- Accepts a user-uploaded image of a single digit.
- Predicts the digit and displays model confidence.
- Shows probabilities for all ten digit classes.
- Beginner-friendly Python code with comments.

## Technologies Used

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pillow (PIL)

## Project Structure

```text
Deep_Learning_Digit_Recognition/
├── app.py                  # Streamlit web app and image preprocessing
├── train.py                # Downloads MNIST, trains and saves the CNN
├── requirements.txt        # Required Python packages
├── README.md               # Project documentation
├── .gitignore
├── LICENSE
└── models/
    └── .gitkeep             # The trained model is created here
```

## How to Run the Project

### 1. Install Python

Use Python 3.10 or 3.11 for best compatibility with the listed TensorFlow versions.

### 2. Download or clone this repository

If you downloaded the ZIP, extract it and open a terminal in the project folder. If using Git:

```bash
git clone https://github.com/YOUR-USERNAME/Deep-Learning-Digit-Recognition.git
cd Deep-Learning-Digit-Recognition
```

Replace `YOUR-USERNAME` with your GitHub username.

### 3. Create and activate a virtual environment (recommended)

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Train the deep learning model

```bash
python train.py
```

The first run downloads the MNIST dataset. Training may take a few minutes depending on your computer. The trained model is saved to `models/digit_cnn.keras`.

### 6. Launch the web app

```bash
streamlit run app.py
```

A local web page will open. Upload a clear image containing one handwritten digit.

## How the Model Works

1. **Dataset:** MNIST contains 28 × 28 grayscale images of handwritten digits.
2. **Preprocessing:** Pixel values are normalized to the range 0–1.
3. **Convolution layers:** Learn visual features such as edges and curves.
4. **Pooling layers:** Reduce the spatial dimensions of feature maps.
5. **Dense layer:** Combines learned features to classify the digit.
6. **Softmax output:** Produces probabilities for the ten possible digits.
7. **Evaluation:** The training script reports accuracy on the test dataset.

## Source Code

The source code is included in this repository as separate files:

- [`train.py`](train.py) — model architecture, training, and evaluation.
- [`app.py`](app.py) — web interface, image preprocessing, and prediction.
- [`requirements.txt`](requirements.txt) — dependencies.

**Note:** The code is intentionally kept in separate `.py` files rather than copied into this README. Open a file in the GitHub repository to view its full source code.

## Expected Result

The app displays the predicted digit, a confidence percentage, and a bar chart of the probabilities for digits 0–9. Actual results depend on the uploaded image and model training. This is an educational project; confidence is not a guarantee that a prediction is correct.

## Limitations

- Best suited to one clearly visible digit per image.
- Unusual handwriting, cluttered backgrounds, or multiple digits may reduce accuracy.
- Image preprocessing is a simple baseline and may need adjustment for some uploaded images.

## Future Improvements

- Add a drawing canvas so users can draw digits directly.
- Improve image centering and thresholding.
- Compare CNN performance with a basic neural network.
- Deploy the Streamlit app online.

## Learning Outcomes

By completing this project, a beginner can learn the basics of image classification, CNN layers, dataset preprocessing, model training, evaluation, and building a small AI application.

## License

This project is available under the MIT License. See [`LICENSE`](LICENSE).
