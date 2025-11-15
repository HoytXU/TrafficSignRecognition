import cv2
import sklearn
import glob
import os
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# T1  start _______________________________________________________________________________
# Read in Dataset

# change the dataset path here according to your folder structure
# Get the directory where this script is located, then go up one level to project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
dataset_path = os.path.join(project_root, "datasets", "dataset1") + os.sep

X = []  # List to store images (features/inputs)
y = []  # List to store class IDs (targets/labels - what we want to predict)
image_files = glob.glob(dataset_path + '*.png', recursive=True)
print(f"Found {len(image_files)} PNG files in {dataset_path}")

for i in image_files:
    # Extract class ID from filename (first 3 characters, e.g., "000", "001", "057")
    # This class ID represents which type of traffic sign this image is
    filename = os.path.basename(i)
    class_id = filename[:3]  # Extract class ID (e.g., "000" from "000_0001.png")
    
    # Read the image and append to X
    img = cv2.imread(i)
    if img is not None:
        X.append(img)
        y.append(class_id)  # Store the class ID for this image
    else:
        print(f"Warning: Could not read image {i}")

print(f"Loaded {len(X)} images")
print(f"Total class IDs stored: {len(y)} (one class ID per image)")
print(f"Number of unique traffic sign classes: {len(set(y))}")
# you should have X, y with 5998 entries on each.
# T1 end ____________________________________________________________________________________


# T2 start __________________________________________________________________________________
# Preprocessing
X_processed = []
for x in X:
    # Write code to resize image x to 48x48 and store in temp_x
    temp_x = cv2.resize(x, (48, 48))
    # Write code to convert temp_x to grayscale
    temp_x = cv2.cvtColor(temp_x, cv2.COLOR_BGR2GRAY)
    # Append the converted image into X_processed
    X_processed.append(temp_x)

# T2 end ____________________________________________________________________________________


# T3 start __________________________________________________________________________________
# Feature extraction
X_features = []
for x in X_processed:
    x_feature = hog(x, orientations=8, pixels_per_cell=(10, 10),
                    cells_per_block=(1, 1), visualize=False)
    X_features.append(x_feature)

# write code to Split training & testing sets using sklearn.model_selection.train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)


#T3 end ____________________________________________________________________________________



#T4 start __________________________________________________________________________________
# Train model
model = SVC()
model.fit(X_train, y_train)

# evaluate model
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# T4 end ____________________________________________________________________________________










