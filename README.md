# Convolutional Neural Network (CNN) for Image Classification

## Contributors:
- **Shefali Pujara (055044)**
- **Vandana Jain (055058)**

## Introduction
This project explores the implementation of Convolutional Neural Networks (CNNs) for image classification. CNNs have significantly advanced the field of computer vision by effectively extracting spatial and hierarchical features from images. The objective is to develop a CNN model that classifies images into predefined categories with high accuracy.

---

## Problem Statement
The primary goal is to construct a robust CNN model for image classification while overcoming challenges such as:

- **Handling variations in lighting conditions, orientations, and background noise**
- **Choosing the optimal CNN architecture and hyperparameters**
- **Reducing overfitting through regularization and data augmentation**
- **Improving classification accuracy for imbalanced datasets**

---

## Dataset Description
The dataset consists of labeled images belonging to two categories. Key details of the dataset:

- **Number of Classes:** 2  
- **Training Set Size:** 60 images  
- **Validation Set Size:** 20 images  
- **Test Set Size:** 20 images  
- **Image Resolution:** Standardized to **128x128 pixels**  
- **Preprocessing Steps:**  
  - Resizing  
  - Normalization  
  - Data Augmentation (Random Rotation, Flipping, Zooming, and Shifting)  

---

## Model Architecture
The CNN model consists of multiple layers to extract relevant features from images. The architecture is structured as follows:

- **Convolutional Layers:** Three layers with 32, 64, and 128 filters to capture different levels of features.
- **Activation Functions:** ReLU is applied after each convolutional layer to introduce non-linearity.
- **Pooling Layers:** Max pooling (2x2) reduces spatial dimensions while preserving important features.
- **Dropout Layers:** Dropout (0.25 and 0.5) prevents overfitting.
- **Fully Connected Layers:** Two dense layers with 128 and 64 neurons for classification.
- **Softmax Output Layer:** Provides probability distributions over the classes.

---

## Implementation Details
The project was implemented using the **Python programming language** with the **TensorFlow and Keras frameworks**.  

- **Programming Language & Framework:** Python (TensorFlow, Keras)
- **Data Augmentation Techniques:** Random rotation, flipping, zooming, and shifting
- **Optimizer:** Adam optimizer with a learning rate of **0.001**
- **Loss Function:** Categorical Cross-Entropy
- **Batch Size:** 32
- **Epochs:** 50

---

## Training & Evaluation
### Training Process:
- The CNN was trained on **60 images**, with validation checks using **20 images**.
- The model's training and validation accuracy were monitored across epochs to assess learning progress.

### Evaluation Metrics:
- **Accuracy**
- **Precision, Recall, and F1-score**
- **Confusion Matrix** for class-wise performance analysis
- **ROC Curve** for classification assessment

### Key Observations:
- **Training accuracy reached 60%**, while **validation accuracy stabilized at 50%**.
- **Overfitting was mitigated** using dropout and data augmentation.
- **Misclassification occurred** in classes with visually similar features.

---

## Results & Discussion
### Final Model Performance:
- **Test Accuracy:** 50%
- **Precision & Recall:** Higher for well-represented classes, lower for underrepresented classes.
- **Misclassification:** Primarily in visually similar categories.
- **Data Augmentation & Hyperparameter Tuning:** Slightly improved performance.

### Insights Gained:
- CNN effectively **captured edges, textures, and shapes** in images.
- Increasing convolutional layers **enhanced pattern recognition**.
- **Class imbalance affected recall**, highlighting the need for balanced datasets.
- Transfer learning (using pre-trained models) **could further improve accuracy**.

---

## Challenges & Future Improvements
### Challenges Faced:
- **Class imbalance** led to biased predictions favoring dominant categories.
- **Noisy and distorted images** caused misclassification.
- **Computational limitations** required optimizing model architecture for efficiency.

### Future Enhancements:
- **Transfer Learning:** Implementing models like ResNet, VGG-16 for better feature extraction.
- **Advanced Optimizers:** Experimenting with RMSprop, AdamW, or learning rate schedules.
- **Hyperparameter Tuning:** Using **Grid Search or Bayesian Optimization** for optimization.
- **Dataset Expansion:** Including more diverse image variations to improve generalization.

---

## Conclusion
This project successfully implemented a CNN model for **image classification**, achieving a test accuracy of **50%**. While the results indicate room for improvement, the model demonstrated an ability to recognize key visual features. Future work will focus on **transfer learning, hyperparameter optimization, and dataset expansion** to enhance classification performance.

---

## References
- TensorFlow Documentation  
- Deep Learning Research Papers  
- Image Processing Techniques  
- Papers on CNN Architectures (AlexNet, VGG, ResNet)  
