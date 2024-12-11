The aim of the project is to create a mapping between clean and augmented images using Gaussian Mixture Models (GMM) and Principal Component Analysis (PCA) for dimensionality reduction. The augmented images are created by introducing noise to clean datasets (e.g., Gaussian, Salt-and-Pepper, and Speckle noises).

Key Features:
Augmentation Techniques:

Introduces Gaussian, Salt-and-Pepper, and Speckle noise to augment the clean images.
These noises simulate real-world image distortions, enhancing the robustness of image recognition tasks.
Dimensionality Reduction Using PCA:

PCA reduces the high-dimensional image data into fewer components, enabling faster and more efficient clustering and mapping.
Clustering via GMM:

Gaussian Mixture Models (GMM) are applied to cluster the reduced data, identifying probable matches between augmented and clean images.
Dataset Construction:

Uses a custom dataset class (AlteredMNIST) to load and process both clean and augmented data from specified directory structures.
Provides mappings between noisy and clean images, stored in a structured dictionary format.
Applications:

Useful in studying the effects of noise on image classification tasks.
Could be extended to tasks requiring robust image recognition in noisy environments.
