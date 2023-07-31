# CodeClause
*Submission project 1 for CodeClause internship 2023 = Color Detection*     
*Submission project 2 for CodeClause internship 2023 = Brain tumor detection*

## Description (color detection)
This project is a simple color detection application that uses K-means clustering to find the dominant colors in an image. The program takes an input image and identifies the most prominent colors present in it. The detected dominant colors are displayed.

## Description (Brain tumor)
This Python script demonstrates image classification using a pre-trained deep learning model. The script uses libraries such as Numpy, Pandas, TensorFlow, and TensorFlow Hub to facilitate the process. It starts by reading image labels from a CSV file and defines functions for image preprocessing, creating data batches for training, validation, and testing. A pre-trained model saved in the ".h5" format is loaded using a custom function. The script then allows users to input custom image file paths, converts them into batches, and uses the loaded model to make predictions on the custom data. The final step involves obtaining the predicted labels for the custom images based on their respective probabilities. Overall, this script provides an efficient and straightforward way to classify images using a pre-trained deep learning model, making it useful for various image classification tasks.

## Installation
Clone the repository or download the project files to your local machine.

> git clone https://github.com/your_username/color-detection-project.git    
> Install the required dependencies using pip.   
> pip install opencv-python numpy scikit-learn matplotlib

## Usage
1. Place your input image in the project directory or provide the full path to the image file.
2. Open a terminal or command prompt and navigate to the project directory.
3. Run the Python script with the desired parameters.
*python color_detection.py path_to_your_image.jpg*
> Replace path_to_your_image.jpg with the actual path to your image file.
4. The script will detect the dominant colors in the image and display it.

## Dependencies
1. OpenCV (opencv-python)
2. NumPy
3. sci-kit-learn
4. Matplotlib

## License
This project is licensed under the MIT License.





