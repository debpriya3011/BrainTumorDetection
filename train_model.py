import os
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.transform import resize

# Function to load the dataset
def load_dataset(image_folder):
    categories = ['notumor', 'glioma', 'meningioma', 'pituitary']
    data = []
    labels = []

    for category in categories:
        folder_path = os.path.join(image_folder, category)
        for img_name in os.listdir(folder_path):
            try:
                img_path = os.path.join(folder_path, img_name)
                img = imread(img_path, as_gray=True)
                img_resized = resize(img, (48, 48))
                data.append(img_resized.flatten())
                labels.append(categories.index(category))
            except Exception as e:
                print(f"Error loading image {img_name}: {e}")
    
    return np.array(data), np.array(labels), categories

# Load dataset
image_folder = "C:/Users/debpr/OneDrive/Desktop/brain_tumour/archive (2)/Training"
X, y, categories = load_dataset(image_folder)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Model training
model = svm.SVC(kernel='linear', C=1)
model.fit(X_train, y_train)

# Save X_train, X_test, y_train, y_test, and categories for use in Code Block 2
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)
np.save('categories.npy', categories)
