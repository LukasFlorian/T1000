#let glossary = (
  Batch: "A batch is a group of data processed together as a unit.",

  SGD: "SGD is an optimization algorithm used to minimize the loss function in machine learning models by iteratively updating the model parameters based on their partial derivatives with respect to individual samples in the training data.",

  Tensor: "A tensor is a mathematical object that generalizes scalars, vectors, and matrices to higher-dimensional arrays.",

  "FC Layer": "A fully connected layer is a layer in a neural network where each neuron is connected to every neuron in the previous layer.",

  CNN: "A convolutional neural network (CNN) is a type of neural network designed to process data with a grid-like topology, such as images.",

  AP: "Average Precision. Calculated as the area under the precision-recall curve.",

  mAP: "Mean Average Precision. Calculated as the mean of the AP values for each class.",

  IoU: "Intersection over Union. A metric used to evaluate the accuracy of object detection models. It is calculated as the ratio of the area of overlap between the predicted bounding box and the ground truth bounding box to the area of union between the two boxes.",

  NMS: "A technique used to eliminate redundant bounding boxes in object detection models. It works by selecting the bounding box with the highest confidence score and eliminating all other bounding boxes that have an IoU greater than a specified threshold with the selected bounding box.",

  SSD: "A single shot multibox detector (SSD) is a type of object detection model that uses a single forward pass of the network to predict the bounding boxes and class scores for all objects in the image.",

  MPS: "Metal Performance Shaders. A framework for accelerating machine learning workloads on Apple Silicon devices.",

  CUDA: "Compute Unified Device Architecture. A parallel computing platform and programming model developed by NVIDIA for general computing on GPUs.",

  ReLU: [Rectified Linear Unit. A type of activation function used in neural networks. It is defined as\ $f(x) = max(0, x)$.],

  ViT: "Vision Transformer. A type of transformer model that is designed for computer vision tasks.",

  YOLO: "You Only Look Once. A type of object detection model that uses a single forward pass of the network to predict the bounding boxes and class scores for all objects in the image.",

  HOG: "Histogram of Oriented Gradients. A feature descriptor used in computer vision and image processing for the purpose of object detection. It is based on the distribution of intensity gradients or edge directions in an image.",

  AdaBoost: "Adaptive Boosting. A type of ensemble learning algorithm that combines multiple weak classifiers to form a strong classifier.",

  SVM: "Support Vector Machine. A type of supervised learning algorithm that is used for classification and regression tasks. It is based on the idea of finding a hyperplane that best separates the data into different classes.",

  "MS COCO": "Microsoft Common Objects in Context. A large-scale object detection, segmentation, and captioning dataset.",

  "PASCAL VOC": "Pascal Visual Object Classes. A dataset for object detection and segmentation tasks.",

  VGP: "Vanishing Gradient Problem. A problem that occurs in deep neural networks where the gradients of the loss function with respect to the weights become very small, making it difficult to train the network.",

  BGD: "Batch Gradient Descent. An optimization algorithm used to minimize the loss function of machine learning models by iteratively updating the model parameters based on their gradients with respect to the entire training dataset.",

  BN: "Batch Normalization. A technique used in deep neural networks to normalize the activations of the layers. It helps to speed up the training of the network and improve its performance.",

  TL: "Transfer Learning. A technique used in machine learning where a pre-trained model is used as the starting point for a new model. It helps to speed up the training of the new model and improve its performance.",

  MBGD: "Mini-Batch Gradient Descent. A type of gradient descent algorithm that updates the weights of the neural network using a small subset of the training data at a time.",

  Backpropagation: "A method used to train neural networks by calculating the gradient of the loss function with respect to the weights of the network and updating the weights in the opposite direction of the gradient. It is called backpropagation because the gradient is calculated from the prediction to the input layers.",

  Sigmoid: [A mathematical function that maps any real number to a value between 0 and 1. It is commonly used as an activation function in neural networks. Refer to #ref(<sigmoid-eqn>) for the mathematical definition.],
)

/*
Tensor? Fully Connected/Linear? Attention? 
*/