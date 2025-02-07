# SpaceCode Challenge 2025
Hosted by the Space Technology and Astronomy Cell (STAC), IIT Mandi

This repository contains the solutions and analyses for the tasks presented in the hackathon. The problems involve analyzing exoplanet data using the Radial Velocity (RV) method, as well as developing machine learning models to classify pulsar signals from radio frequency interference (RFI).

## Table of Contents
1. [Problem 01: Analysis of Exoplanets Based on Orbital Time Period](#problem-01-analysis-of-exoplanets-based-on-orbital-time-period)
2. [Problem 02: Radial Velocity Method for Finding Exoplanets](#problem-02-radial-velocity-method-for-finding-exoplanets)
3. [Problem 03: Pulsars in Deep Space](#problem-03-pulsars-in-deep-space)
4. [Theoretical Questions](#theoretical-questions)
5. [Technologies Used](#technologies-used)
6. [Setup Instructions](#setup-instructions)

## Problem 01: Analysis of Exoplanets Based on Orbital Time Period

The goal of this problem was to analyze a catalog of exoplanets discovered using the Radial Velocity method. The main task was to create a histogram plot of the number of exoplanets as a function of their orbital period around their host stars.

### Tasks
- **Dataset**: Exoplanet data available from [Exoplanet.eu](https://exoplanet.eu/catalog/).
- **Objectives**:
  - Generate a histogram plot showing the number of exoplanets discovered as a function of their orbital time period.
  - Analyze and provide insights into why most exoplanets exhibit short orbital periods (less than 50 Earth days).
  - Discuss the challenges of detecting exoplanets with longer orbital periods, such as Earth-like planets.

### Deliverables:
- Histogram plot.
- Detailed analysis addressing the questions on short-period exoplanets and the challenges of detecting longer orbital periods.

## Problem 02: Radial Velocity Method for Finding Exoplanets

In this problem, we worked with both synthetic and real radial velocity (RV) data to simulate and analyze the dynamics of star-exoplanet systems. Specifically, we focused on the first Sun-like star, 51 Pegasi, which hosted an exoplanet detected using the RV method.

### Tasks
- **Dataset**: Radial velocity data for the star 51 Pegasi can be found [here](https://drive.google.com/file/d/1fOckX-ElhDkeRA2xOyb0mxUF23lgbP3_/view).
- **Objectives**:
  1. Create a plot of the radial velocity data against time.
  2. Estimate the orbital period of the exoplanet from the scatter plot.
  3. Perform a Lomb-Scargle periodogram analysis to rigorously estimate the orbital period.
  4. Fold the radial velocity data based on the estimated orbital period and plot the folded curve.

### Deliverables:
- Scatter plot of time vs. radial velocity.
- Initial orbital period estimation from the scatter plot.
- Lomb-Scargle periodogram analysis and explanation.
- Folded radial velocity curve.

## Problem 03: Pulsars in Deep Space

This problem focused on classifying pulsar signals from man-made radio frequency interference (RFI) using two datasets: one with numerical features and one with images representing raw signal data.

### Reports

The following PDF file contains the detailed report for **Problem 3: Pulsar Classification**:

- [Problem 03: Pulsar Classification Report](CelestialCoders_SCC_Problem3/CelestialCoders_SCC_Q3.pdf)

## Theoretical Questions

The following PDF files contain the handwritten solutions for the theoretical questions solved by hand:

- [Problem 04: Orbital Resonances](CelestialCoders_SCC_Problem4/Question%204.pdf)
- [Problem 05: Escape Velocity from a White Dwarf](CelestialCoders_SCC_Problem5/Question%205.pdf)

### Tasks
- **Datasets**:
  - **Numerical Features Dataset**: Contains eight numerical features per sample, suitable for traditional machine learning models such as Random Forests, SVMs, and neural networks. [Dataset Links: Training](https://t.ly/6WL6Q), [Testing](https://t.ly/z6Co_).
  - **Image Dataset**: A dataset of images representing raw signal data, suitable for convolutional neural networks (CNNs). [Image Dataset Link](https://as595.github.io/HTRU1/).

- **Objectives**:
  1. Preprocess both Numerical and Image Datasets, including handling missing values, scaling, and transformations for CNN models.
  2. Develop and evaluate machine learning models (Random Forest, SVM, Neural Networks) for the numerical dataset and CNNs for the image dataset.
  3. Evaluate model performance using metrics like accuracy, precision, recall, F1 score, and ROC-AUC.
  4. Compare and discuss the performance of models trained on the numerical dataset versus the image dataset.

### Deliverables:
- Preprocessed datasets and code for Data Cleaning and Transformations.
- Well-documented code for all models.
- Performance evaluation report, including confusion matrices, ROC curves, and feature importance.

## Technologies Used
- Python
- Libraries: `scikit-learn`, `PyTorch`, `TensorFlow`, `matplotlib`, `seaborn`, `SciPy`
- Tools: Jupyter Notebooks for Code Development

## Setup Instructions

1. Clone the Repository:
    ```bash
    git clone https://github.com/Dev31415926535/Space-Code-Challenge.git
    cd Space-Code-Challenge
    ```

2. Install dependencies:
   Make sure you have all the required libraries and dependencies installed. Run the following command:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn torch  torchvision
   ```
