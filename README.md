This repository is for the project of the course Computer Vision (COSC 444/544).
# FireFinder- A Feature Centric Pipeline for Robust Fire Detection in Images

This repository contains the code, dataset, and pipeline for a pixel-wise fire detection system using handcrafted features. The approach combines color features from multiple color spaces (e.g., RGB, HSV, Lab, YCrCb) with Gabor-based texture features to build a compact and effective feature representation for fire classification.

We evaluated the model using logistic regression and LightGBM classifiers. Our selected feature set outperformed models trained on individual color spaces across all key metrics—including accuracy, precision, recall, and false positive rate—achieving up to 90.68% accuracy with LightGBM. The code includes preprocessing steps, feature extraction, model training, and performance evaluation, as well as experiments on out-of-distribution images.

### Summary of pipeline
- Color channel extraction from RGB, HSV, Lab, and YCrCb
- Gabor filter bank for texture representation
- Feature compression (extraction) using the Komologonov Smirnoff Test.
- Pixel-level classification using logistic regression and LightGBM
- Evaluation on both in-distribution and out-of-distribution samples

### Structure
to be completed


#### Team members:
1. Ladan Tazik
2. Beck Corkle
3. Jacob Tizel
4. Seth Richard
5. Zoë Dale

🔗 Citation
If you use this code in your research or project, please cite this repository.
