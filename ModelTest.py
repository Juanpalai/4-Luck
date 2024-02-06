import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the segmentation model
segmentation_model = load_model('four_leaf_clover_seeker.h5')

def plot_images(test_image, save_path=None):
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 3, 1)
    plt.imshow(test_image)
    plt.title('Image')
    if save_path:
            plt.savefig(save_path, format='jpg')
    else:
        # Show the plot if save_path is not provided
        plt.show()

def postprocess_predictions(predictions, threshold=0.5):
    binary_mask = (predictions > threshold).astype(np.uint8)
    if binary_mask.shape[-1] == 1:
        binary_mask = np.squeeze(binary_mask, axis=-1)
    return binary_mask

def preprocess_image(image):
    resized_image = cv2.resize(image, (256, 256)) / 255.0
    return resized_image

def predict_test_samples(img):
    test_images = np.array([img])
    predictions = segmentation_model.predict(test_images)
    return predictions

# Open the camera
capture = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = capture.read()

    # Preprocess the frame
    preprocessed_image = preprocess_image(frame)

    # Get predictions from the segmentation model
    predictions = predict_test_samples(preprocessed_image)

    # Postprocess the predictions
    segmented_mask = postprocess_predictions(predictions)

    # Convert BGR to RGB for displaying
    frame_rgb = cv2.cvtColor(segmented_mask, cv2.COLOR_BGR2RGB)

    # Display the original frame and the segmented mask side by side
    #cv2.imshow('Camera', frame_rgb)
    plot_images(frame_rgb, save_path='frame_rgb.jpg')
    # plot_images(predictions)
    # plot_images(segmented_mask)
    # plot_images(frame_rgb)
    break

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
capture.release()
cv2.destroyAllWindows()
