# ISL_Sign_converter
A simple sign language recognizer using SVM and scikit-learn's adjusted_rand_score (basic correlation).


Steps to setup before you run:
1) install pandas package (pip3 install pandas)
2) install sklearn (pip3 install scikit-learn)
3) Create a folder for 'images' and subfolders for 'TRAIN' and 'TEST' images.
4) Create a folder 'data'.
5) run generate_image_features.py
6) run train.py

Now run the main.py file.

Folder name convention for training set:
images/train/A (contains all the photos for A's symbol,similar for other characters)
Folder name convention for test set:
images/train/A.jpg (Test for A's symbol,similar for other characters)

