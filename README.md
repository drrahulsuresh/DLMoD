# DLMoD
The models.py file defines a set of advanced deep learning models, including ResNet, DenseNet, MIL, VIT, and CapsuleNet, all inheriting from a unified AdvancedSetup base class. Each model architecture is configured with a customizable forward pass, while AdvancedSetup provides shared, advanced training and validation functions.

This repository provides a collection of advanced deep learning models, including ResNet, DenseNet, Multiple Instance Learning (MIL), Vision Transformer (VIT), and CapsuleNet, all built with an extensible AdvancedSetup base class. Each model benefits from a consistent training setup with advanced features like per-epoch metric tracking, flexible pooling, and checkpointing.

Features
Advanced Training and Validation: Models inherit shared training and validation functions with extensive metric logging, flexible pooling methods, and checkpointing capabilities.
Modular Design: Each model (ResNet, DenseNet, MIL, VIT, CapsuleNet) is defined separately for easy customization, while inheriting advanced training features from a single base class.
Support for Instance and Bag Labels: Especially for MIL, this repository includes support for both instance and bag-level labels, useful for complex image and text classification tasks.

git clone https://github.com/your-username/advanced-deep-learning-models.git
cd advanced-deep-learning-models
pip install -r requirements.txt


Model Descriptions
ResNetModel: Implements ResNet with customizable configurations (e.g., ResNet18, ResNet50).
DenseNetModel: Uses DenseNet121 architecture with an adjustable classifier layer.
MILModel: A Multiple Instance Learning model that supports both mean and max pooling for bag aggregation.
VITModel: A Vision Transformer model with token pooling.
CapsuleNet: A Capsule Network model with primary capsules and a fully connected classifier.
