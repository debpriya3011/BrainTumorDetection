# BrainTumorDetection
Check tumor is present in a given image or not using machine learning

Brain Tumor Classification GUI
This repository contains a GUI application for classifying brain tumors from MRI images. The application uses a pre-trained Support Vector Machine (SVM) model to predict the type of brain tumor (No Tumor, Meningioma, Glioma, or Pituitary) from an input image.

Features
Image Upload: Browse and select an MRI image for classification.
Image Display: View the selected image within the GUI.
Prediction: The application displays the predicted class (tumor type) within the GUI.
Save Prediction: Option to save the predicted tumor type to a pre-saved prescription.txt file.
Requirements
Python 3.x
Libraries:
numpy
matplotlib
scikit-learn
scikit-image
Pillow
tkinter

 You can install the required libraries using the following command:

 ![image](https://github.com/user-attachments/assets/0adc5901-34d8-49fc-8946-2115cc684cfc)

 Step-by-Step Explanation:
**Step 1: Import Necessary Libraries**
![image](https://github.com/user-attachments/assets/d8ba7104-c5fb-4096-9b7b-247c3191da42)


•	 os: Used for interacting with the operating system, especially for file and directory manipulation.
•	numpy (np): Used for numerical operations on arrays.
•	sklearn: Contains the svm module for Support Vector Machine modeling and train_test_split for splitting the dataset.
•	skimage: Used for image processing, specifically imread for reading images and resize for resizing images.
**Step 2: Define Function to Load the Dataset**
![image](https://github.com/user-attachments/assets/9616c894-101d-449b-adec-ee5744541f21)

 
•	categories: A list of the different categories of brain tumors and the 'notumor' category.
•	data: An empty list to store the image data.
•	labels: An empty list to store the corresponding labels for the images.
•	for category in categories: Iterates over each category to load the images.
o	folder_path: Constructs the path to the folder containing images of the current category.
o	for img_name in os.listdir(folder_path): Iterates over each image in the category folder.
	img_path: Constructs the full path to the image.
	img = imread(img_path, as_gray=True): Reads the image as a grayscale image.
	img_resized = resize(img, (48, 48)): Resizes the image to 48x48 pixels.
	data.append(img_resized.flatten()): Flattens the 48x48 image into a 1D array and adds it to the data list.
	labels.append(categories.index(category)): Adds the index of the current category to the labels list.
o	except Exception as e: Catches any errors that occur during image loading and prints an error message.
•	return np.array(data), np.array(labels), categories: Converts the data and labels lists to numpy arrays and returns them along with the categories.
**Step 3: Load the Dataset**

![image](https://github.com/user-attachments/assets/d19cdce7-534b-4c51-9e93-cc2648e0b7b3)

 image_folder: Path to the folder containing the training images.
•	X, y, categories: Calls the load_dataset function to load the image data and labels, and assigns the output to X (data), y (labels), and categories (category names).
**Step 4: Split the Dataset into Training and Testing Sets**
![image](https://github.com/user-attachments/assets/20cb169e-04cd-4f14-a536-5023b5e88e57)

 
•	train_test_split: Splits the data into training and testing sets.
o	X_train: Training data.
o	X_test: Testing data.
o	y_train: Training labels.
o	y_test: Testing labels.
o	test_size=0.1: 10% of the data is used for testing.
o	random_state=42: Ensures reproducibility of the split by setting a seed for random number generation.

**Step 5: Train the Model**

![image](https://github.com/user-attachments/assets/8d323f10-44e0-444b-997b-620e4cf92a9d)


•	 model = svm.SVC(kernel='linear', C=1): Initializes an SVM model with a linear kernel and a regularization parameter C set to 1.
•	model.fit(X_train, y_train): Trains the SVM model using the training data and labels.
**Step 6: Save the Data and Model for Future Use**
![image](https://github.com/user-attachments/assets/ea41484f-dae2-4710-9d1a-4bdfc75e48d8)

•	 np.save: Saves the numpy arrays to files with .npy extension.
o	'X_train.npy': File to store the training data.
o	'X_test.npy': File to store the testing data.
o	'y_train.npy': File to store the training labels.
o	'y_test.npy': File to store the testing labels.
o	'categories.npy': File to store the category names.
**Summary**
1.	Import Libraries: Import necessary libraries for file handling, image processing, and machine learning.
2.	Define Dataset Loading Function: Define a function to load images, preprocess them, and assign labels.
3.	Load Dataset: Load the dataset from the specified folder.
4.	Train-Test Split: Split the data into training and testing sets.
5.	Train the Model: Initialize and train an SVM model.
6.	Save Data and Model: Save the preprocessed data and category information for later use.





