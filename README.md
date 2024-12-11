Overview
This program performs dataset augmentation and mapping of augmented images to clean images using a combination of dimensionality reduction (PCA) and Gaussian Mixture Models (GMM). It is designed to enhance datasets for machine learning applications and ensures a robust mapping between augmented and clean datasets.

Features
Dataset Loading: Loads clean and augmented datasets from specified directories.
Data Augmentation: Adds noise (Gaussian, salt-and-pepper, and speckle) to images for better model generalization.
Dimensionality Reduction: Uses Principal Component Analysis (PCA) to reduce image dimensionality.
Gaussian Mixture Models (GMM): Fits GMMs for clean images and creates mappings for augmented images.
Mapping Creation: Maps each augmented image to its closest clean image using the GMM models.
Output Saving: Saves the augmented-to-clean mapping in a mapping.pickle file.
Requirements
Python 3.7 or later
Required libraries:
torch
torchvision
numpy
scikit-learn
pickle
os
re
Install the required Python libraries using:

bash
Copy code
pip install torch torchvision numpy scikit-learn
Usage
Setup Directory Structure:

Place the clean dataset in ../DLA3/Data/clean/.
Place the augmented dataset in ../DLA3/Data/aug/.
Run the Program: Execute the script using:

bash
Copy code
python program.py
Output:

Augmented images with noise are saved in the respective directories.
Augmented-to-clean mapping is saved as a pickle file: mapping.pickle.
Code Workflow
Dataset Loading:

Loads clean and augmented datasets using torchvision.datasets.ImageFolder with appropriate transformations.
Noise Augmentation:

Applies Gaussian, salt-and-pepper, and speckle noise to the images to create a variety of augmented samples.
Dimensionality Reduction:

Uses PCA with 128 components to reduce the dimensionality of images for computational efficiency.
Fitting GMM:

Fits a Gaussian Mixture Model for each label (0–9) using the PCA-transformed clean dataset.
Mapping Augmented to Clean:

Maps each augmented image to its closest clean image by calculating probabilities using the GMM.
Save Results:

Saves the augmented-to-clean mapping in a pickle file.
Directory Structure
bash
Copy code
.
├── program.py           # Main script
├── ../DLA3/Data/        # Parent data directory
│   ├── clean/           # Directory containing clean images
│   ├── aug/             # Directory containing augmented images
├── mapping.pickle       # Output file containing augmented-to-clean mapping
Customizing Parameters
Modify PCA components:
python
Copy code
pca = PCA(n_components=128)
Change GMM components:
python
Copy code
gmm = GaussianMixture(n_components=50)
Expected Outputs
mapping.pickle: Contains a dictionary mapping each augmented image to its closest clean image.
Augmented images with noise saved in directories corresponding to their labels.
Limitations
Assumes datasets are pre-organized in a specific structure.
Computationally intensive for large datasets due to GMM fitting.
Contact
For questions or improvements, please contact:

Author: Udit
