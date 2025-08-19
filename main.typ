#import "@preview/supercharged-dhbw:3.4.1": *
#import "acronyms.typ": acronyms
#import "glossary.typ": glossary

#set par(spacing: 1.5em)
#show list: set block(spacing: 1.5em)

#let abstract = [
  This project report evaluates the performance of neural object detection models for detecting humans in infrared images. The study focuses on comparing different variations of the SSD (Single Shot Mutlibox Detector) model architecture, assessing their accuracy and inference speed, and identifying the most suitable model for the given task. Additionally, different preprocessing techniques are evaluated to improve the detection performance.

  More specifically, the main contributions of this project are:
  - Conceptualization of a simple and cost-efficient hardware setup for the purpose of on-premise human detection in infrared images
  - Evaluation of different SSD model architectures
  - Comparison between different preprocessing techniques
  - Identification of the most suitable model for the given task
  - A theoretical pipeline for the secure transmission of the detection results to a remote server
]

#show: supercharged-dhbw.with(
  title: "Evaluation of Neural Object Detection Models for Human Detection in Infrared Images",
  authors: (
    (name: "Lukas Florian Richter", student-id: "None", course: "TIK24", course-of-studies: "Computer Science - Artificial Intelligence", company: (
      (name: "Airbus Defence & Space", city: "Taufkirchen")
    )),
  ),
  acronyms: acronyms, // displays the acronyms defined in the acronyms dictionary
  at-university: false, // if true the company name on the title page and the confidentiality statement are hidden
  bibliography: bibliography("sources.bib"),
  date: datetime.today(),
  glossary: glossary, // displays the glossary terms defined in the glossary dictionary
  language: "en", // en, de
  supervisor: (company: "René Loeneke"),
  university: "Cooperative State University Baden-Württemberg",
  university-location: "Ravensburg Campus Friedrichshafen",
  university-short: "DHBW",
  // for more options check the package documentation (https://typst.app/universe/package/supercharged-dhbw),
  type-of-thesis: "PROJECT REPORT T1000",
  logo-right: image("./assets/AIRBUS_Blue.png"),
  //logo-left: image("./assets/DHBW_Logo.png"),
  logo-size-ratio: "2:1",
  header: (
    display: true,
    show-chapter: true,
    show-left-logo: false,
    show-right-logo: true,
  ),
  time-to-complete: "16 Wochen",
  abstract: abstract,
  show-confidentiality-statement: false,
)


// Table of Contents and Content Structure

= Introduction <intro>

With the increase in security threats to critical infrastructure, automated surveillance systems have become essential for ensuring the safety and security of people, infrastructure and property at scale. The ability to detect individuals on critical infrastructure premises is crucial to preventing unauthorized access and potential damage to assets. While conventional RGB-based surveillance systems remain prevalent in many application, they face inherent limitations in challenging scenarios such as low-light conditions, adverse weather, fog, smoke and complete darkness during nighttime @farooqObjectDetectionThermal2021.

A compelling alternative to systems operating in the visible light domain are such capturing wavelengths in the infrared spectrum and thus offering consistent detection capabilities that are fundamentally independent of ambient lighting conditions. Unlike regular RGB cameras, thermal sensors detect light with longer wavelengths that correspond to the heat signatures emitted directly by objects. This characteristic provides unique advantages for human detection, as the human body maintains a relatively constant temperature of approximately 37°C, creating distinct thermal signatures that remain visible regardless of environmental illumination @akshathaHumanDetectionAerial2022.

The integration of deep learning architectures with thermal imaging thus opens new possibilities for automated systems that can reliably detect humans in the aforementioned scenarios with adverse conditions for conventional RGB-based concepts. However, most state-of-the-art object detection models have been primarily developed for and trained on RGB imagery. Given that the spectral, tectural and contrast characteristics of infrared images differ substantially from visible-light imagery, both due to the properties of those wavelengths themselves and of the sensors, those existing models might need to be adapted to achieve optimal performance.

//
This research addresses the critical need for systematic evaluation of neural object detection models specifically tailored for thermal human detection applications. The study focuses on the #acr("SSD") architecture, a prominent one-stage detection framework known for its balance between accuracy and computational efficiency. By examining multiple model variants with different backbone networks (VGG16 and ResNet152), initialization strategies (pretrained versus scratch training), and thermal-specific preprocessing techniques (image inversion and edge enhancement), this work provides comprehensive insights into optimal configurations for infrared surveillance systems.

/*
This project addresses the need for systematic evaluation of neural object detection models and preprocessing techniques tailored for thermal human detection applications. For the purpose of developing and edge-deployable network, the focus of this study lies on the #acr("SSD") architecture, a prominent one-stage detection framework with relatively low complexity compared to newer architectures like the #acr("ViT").
*/

/*
This work provides comprehensive insights into optimal configurations for infrared surveillance networks by examining multiple model variants with different:
+ *backbone networks* (#acr("VGG") and #acr("ResNet"))
+ *initialization strategies* (parameters pretrained on the RGB-Dataset IMAGENET1K_V2 versus randomly sampled)
+ *preprocessing techniques* (inversion of the image or enhancement of its edges)
*/

The practical significance of this research extends beyond academic interest, addressing real-world challenges faced by the security and defense industry. In partnership with Airbus Defence & Space, this project explores the development of cost-efficient, edge-deployable thermal surveillance solutions that can operate reliably in challenging environments where traditional RGB systems fail.

== Research Objectives and Contributions

This thesis makes several key contributions to the field of thermal image processing and computer vision:

+ *Preprocessing Technique Analysis*: Quantitative evaluation of thermal-specific image enhancement methods, including polarity inversion and edge enhancement, and their impact on detection accuracy.

+ *Backbone Network Comparison*: Detailed comparison between VGG16 and ResNet152 architectures in the context of thermal imagery, addressing the trade-offs between model complexity and performance.

+ *Practical Implementation Guidelines*: Development of actionable recommendations for deploying thermal surveillance systems in real-world environments, considering computational constraints and accuracy requirements.

+ *Dataset Integration Framework*: Unified evaluation approach across five diverse thermal datasets (FLIR ADAS v2 @FREEFLIRThermal, AAU-PD-T @hudaEffectDiverseDataset2020, OSU-T @davisTwoStageTemplateApproach2005, M3FD @liuTargetawareDualAdversarial2022, KAIST-CVPR15 @hwangMultispectralPedestrianDetection2015), enabling robust performance assessment.

== Thesis Organization

The remainder of this thesis is structured to provide a comprehensive examination of thermal human detection using neural networks. @literature presents a thorough review of object detection fundamentals, SSD architecture principles, and thermal image processing techniques, establishing the theoretical foundation for the experimental work. @methodology details the systematic approach employed for model evaluation, including dataset preparation, experimental design, and evaluation metrics. @results presents comprehensive performance analysis across all model configurations and preprocessing techniques. @discussion interprets the findings within the context of practical deployment scenarios and industrial requirements. Finally, @conclusion synthesizes the key contributions and outlines directions for future research in thermal surveillance technologies.


/*
This research represents a significant step toward practical implementation of AI-powered thermal surveillance systems, providing the empirical foundation necessary for informed decision-making in security-critical applications where reliable human detection is paramount.
*/

= Literature Review and Theoretical Background <literature>

The field of object detection has undergone significant evolution from traditional computer vision techniques to sophisticated deep learning architectures. Understanding this progression is essential for contextualizing the current work's contribution to thermal image analysis. This section examines the theoretical foundations of object detection, with particular emphasis on the Single Shot MultiBox Detector (SSD) architecture and its applicability to thermal imagery processing challenges.

*Key areas to develop:*
- Evolution from traditional methods (HOG, SIFT) to deep learning
- Comparison of one-stage vs. two-stage detection models
- SSD architecture fundamentals and anchor box mechanisms
- Backbone network analysis (VGG vs. ResNet trade-offs)
- Thermal imaging characteristics and preprocessing challenges
- Existing work on infrared human detection
- Gap analysis: Limited research on SSD for thermal surveillance

== Object Detection Fundamentals <obj-detection>
Most object detection methods can be broadly categorized into two main approaches: traditional methods and deep learning-based methods. Traditional methods mainly rely on handcrafted features and sliding window techniques, while deep learning-based methods in this field leverage #acrpl("CNN") or #acr("ViT") architectures to automatically learn features from data.

=== Traditional Object Detection Methods <traditional-methods>
Simple approaches to object detection entail applying manually constructed feature detector kernels in a sliding window fashion to images. 

An example of this is the *Viola-Jones-Algorithm* @violaRapidObjectDetection2001:
+ Compute the integral image of the input image, that is, the sum of pixel intensities from the top-left corner of the image to each pixel. This allows for quick computation of the sum of pixel intensities in any rectangle in the image by subtracting the value of the upper left pixel of the rectangle from that of the lower right pixel.

+ Apply a series of Haar-like features to detect potential objects. Haar-like features are computed by subtracting the sum of pixels in one rectangle from the sum of pixels in an adjacent rectangle. These features capture various simple patterns, such as edges and lines.

+ Use the #acr("AdaBoost") technique to build a cascaded strong classifier consisting of several weak classifiers that can detect simple patterns consisting of Haar-like features.

+ Split the image into subwindows and classify each subwindow using the cascaded classifier as either containing the object or not.

Other approaches employ #acr("HOG") descriptors. The #acrpl("HOG") are attained by dividing the image into a grid of cells, contrast-normalizing them and then computing the vertical as well as horizontal gradients of their pixels. The gradients for each cell are accumulated in a one-dimensional histogram which serves as that cell's feature vector. After labeling the cells in the training data, a #acr("SVM") can be trained to find an optimal hyperplane separating the feature vectors corresponding to the object that should be detected from those that do not contain the object.

=== Deep Learning-Based Object Detection <deep-learning-detection>
However, those methods are either highly dependent on engineering the correct priors, such as the Haar-like features, or limited to binary classification scenarios, as is the case for #acr("HOG")-based #acrpl("SVM"). Thus, newer Object Detection methods employ more complex deep-learning architectures that require less manual feature engineering. The best-performing models nowadays are #acrpl("ViT") using Attention mechanisms @dosovitskiyImageWorth16x162021 to learn relationships between patterns in different parts of images. However, they will not be further examined in this thesis, due to computational constraints that make them unfeasible for the edge-deployable solution sought in this work @dosovitskiyImageWorth16x162021.

Relevant for this examination are their predecessors, #acrpl("CNN"). The main mechanism they use to extract information from images are convolutional layers. Those convolutional layers get passed an image in the form of a tensor and perform matrix multiplication on that input tensor and a kernel tensor in a sliding window fashion to compute subsequent feature maps. Those will be passed on as input to the next layer. @lecunHandwrittenDigitRecognition1989

At their core, these convolutional layers do not work inherently different from fully connected layers that compute several weighted sums across all components of the input tensor. More specifically, fully connected layers can be described as convolutional layers whose kernel dimensions are identical to those of the input tensor.

Resorting to smaller kernels, however, serves as a prior making use of the heuristic that in most cases, the features composing an object in an image lie closely together. Thus, it is not necessary to process the entire image to detect an object that occupies only part of it. Convolutional neural nets hence save computational resources by focusing on smaller regions. In many cases it is advantageous to use those savings to increase network depth in order to make it possible for the network to learn more complex high-level features in subsequent layers.

Object detection, as opposed to image classification, consists of two main tasks: locating where an object is and classifying which class it belongs to. In the context of machine learning, that means two concepts must be used: regression to approximate the location of an object and classification to determine its class. #acrpl("CNN") solving these tasks can be categorized into two main categories:

- *Two-Stage Detectors*: These detectors operate in two stages. The first stage proposes regions of interest and the second stage classifies which object they contain. In more detail, that means regressing bounding boxes and assessing the "objectness" of that region, for example by using logistic regression. If the confidence this region contains an object exceeds a given threshold, the second stage then classifies the object in that region. That requires a second pass of the extracted region through a classifier network. This two-stage approach can be computationally expensive, especially when dealing with a large number of proposals. Examples of two-stage detectors include #acrpl("R-CNN") @girshickRichFeatureHierarchies2014, Fast #acrpl("R-CNN") @girshickFastRCNN2015, and Faster #acrpl("R-CNN") @renFasterRCNNRealTime2016.

- *Single-Stage Detectors*: These detectors perform both tasks simultaneously in a single pass through the network. That means passing the image through a network that both regresses bounding boxes and classifies objects in those boxes at the same time. Examples include #acr("YOLO") @redmonYouOnlyLook2016  and #acr("SSD") @liuSSDSingleShot2016. This approach can be faster but may sacrifice some accuracy compared to two-stage detectors, as the feature extractor is not optimized for both tasks .

Given the computational constraints imposed by the requirement for edge-deployment, single-stage detectors were chosen. Past research has shown that #acr("SSD")-variants with Inception-v2 and MobileNet-v1 backbones perform notably faster than their Faster #acr("R-CNN") counterparts, namely 4 to 7 times as fast @akshathaHumanDetectionAerial2022.

Furthermore, benchmarks of #acr("SSD") and #acr("YOLO") on the MS COCO dataset yielded similar results favoring #acr("SSD") in terms of speed when deployed on edge devices, namely the Raspberry Pi 4 both with and without a dedicated #acr("TPU") @alqahtaniBenchmarkingDeepLearning2024. #acr("YOLO") did deliver higher #acr("mAP") scores, but the difference was not significant enough to justify the trade-off in speed, in particular taking into account the benefit of speed for real-time applications when multiple images are captured each second and fast enough processing allows for multiple attempts at detection. Additionally, the #acr("SSD") models tested consumed less energy than their #acr("YOLO") counterparts @alqahtaniBenchmarkingDeepLearning2024, making them a more suitable choice that minimizes the need for human intervention to replace the battery of the edge device, which is a significant factor in the cost of deployment and maintenance of the system.

== #acr("SGD") as Optimizer in Deep Learning <sgd-optimizer>
Deep Learning Models are optimized by minimizing the loss function $L(accent(y, hat), y)$, which is a measure of the difference between the predicted output $accent(y, hat)$ and the expected output $y$, i.e. the ground truth. The loss function is typically a differentiable function, which means that it can be used to compute the gradient of the loss with respect to the model parameters. The gradient is then used to update the model parameters in the direction that minimizes the loss function, hence the name gradient descent.

This happens in reverse order of the forward pass, since gradients of the loss function w.r.t. earlier layer's parameters are computed using the chain rule, which is why this process is also called backpropagation. More specifically, the algorithm:
+ First computes the gradient of the loss function w.r.t. a layer's output
+ Then computes the gradient of the loss function w.r.t. the layer's parameters using the chain rule.
+ updates the layer parameters are updated using the gradient computed in the second step
+ Now the algorithm moves on to the next (earlier) layer and repeats the process until it reaches the first layer of the network.
The challenges posed by this process are discussed in @vgd


In most cases, the training dataset is too large to compute the gradient with respect to all data at once. Instead, the optimization takes place in so-called mini-batches of training data of a fixed size, under the assumption that the gradient computed with respect to a mini-batch of training data is a good approximation of the gradient that would be obtained if the calculation was performed across the entire training dataset.

This differentiation of the loss function with respect to a mini-batch of training data is called #acrl("SGD"), (#acr("SGD")).

As desribed before, #acr("SGD") is a first-order optimization algorithm, which means that it only considers the first-order derivatives of the loss function with respect to the model parameters. This is in contrast to second-order optimization algorithms, such as Newton's method, which consider the second-order derivatives of the loss function with respect to the model parameters. However, first-order optimization algorithms are generally preferred in deep learning because they are computationally more efficient and can be easily implemented on hardware accelerators such as #acrpl("GPU") and #acrpl("TPU"), which is why second-order optimization algorithms will not be further discussed in this work.

=== The Vanishing Gradient Problem <vgd>

One limitation of #acr("SGD") lies in the so-called #acr("VGP"), which occurs due to the calculation of the partial derivates by means of the chain rule. The chain rule for differentiation states that the derivative of a composite function is the product of the derivatives of its components:

$ 
(f(g(x)))' = f'(g(x)) dot g'(x)
$

In this equation, $f$ and $g$ are the functions applying the linear transformations corresponding to a convolutional layer, its successor layer and their activation functions, respectively. Since both convolutional kernels contain a large number of parameters, each individual parameter only has a small impact on the overall loss function. As a result, the product of the derivatives of the loss function with respect to the parameters of the two layers can become very small, leading to a phenomenon known as the #acr("VGP").

In the context of deep learning, this means that deeper models are particularly likely to suffer from this problem, as more layers amplify the core issue. One approach to mitigate this problem is to introduce the so-called residual layers allowing for an identity mapping of the input to the output of a layer, which is a key component of the #acr("ResNet") architecture.

=== Residual Layers to Mitigate the Vanishing Gradient Problem <resnet-arch>

Residual layers are a key component of the #acr("ResNet") as well as other modern architectures, and are designed to mitigate the #acr("VGP") problem. The idea behind residual layers is to introduce a shortcut connection that allows the input to the layer to be added to the output of the layer. This shortcut connection allows the gradient to "flow directly through the layer", which helps to prevent the gradient from vanishing as it is backpropagated through the network. The following equation shows the residual layer:
$
  accent(y, hat) = F(x) + x
$
where $F$ is the residual function, $x$ is the input to the layer, and $accent(y, hat)$ is the output of the layer. The residual function $F$ is typically a stack of convolutional layers, #acr("BN") layers, and #acr("ReLU") activation functions. @heDeepResidualLearning2015

If the function $F$ is a composite function $F(x) = f(g(x))$, then the derivative of $accent(y, hat)$ with respect to $x$ is given by the chain rule:
$
  accent(y, hat)'(x) &= F'(x) + x'\
  &= f'(g(x)) dot g'(x) + 1
$
Here, the derivative of $x$ with respect to itself is 1. In reality, $x$ would be the result of a previous layer's function $h(x_0)$, which implies for the derivative of $accent(y, hat)$ with respect to $x_0$:
$
  accent(y, hat)(x)' &= accent(y, hat)(h(x_0))'\
  &= accent(y, hat)'(h(x_0)) dot h'(x_0)\
  &= (F'(h(x_0)) + 1) dot h'(x_0)\
  &= f'(g(h(x_0))) dot g'(h(x_0)) dot h'(x_0) + h'(x_0)
$
It becomes apparent that introducing residual layers and identity shortcuts helps to mitigate the #acr("VGP") by preserving the gradient w.r.t. previous layers' outputs without multiplying them with many small derivatives of subsequent layers. $F(x)$ can be an arbitrarily deep function and the gradient w.r.t. $x_0$ will still be $h'(x_0)$ plus a term that is multiplied by the derivatives of the residual function $F(x)$. This is in contrast to the gradient in a standard #acr("CNN") where repeated application of the chain rule is likely to cause the gradient to vanish.

When the input and output dimensions of a residual layer do not match, a linear projection is used to match the dimensions of the input and output before adding the two together. This linear projection is typically implemented using a 1x1 convolutional layer @heDeepResidualLearning2015. Even though this practice introduces additional parameters that need to be learned and makes the identity shortcut actually not an identity function anymore, it is still beneficial to the training process, as the residual function can be used to significantly increase the network's depth while preserving the gradient flow.

*TODO ONCE TEXT IS DONE: Residual Layer visualization*

== Single Shot MultiBox Detector (SSD) Architecture <ssd-arch>
The SSD architecture is a single-stage detector that uses a base network to extract features from the input image and then applies additional convolutional layers as well as #acr("FC") layers to predict bounding boxes and class scores for each feature map. The following sections provide a more detailed explanation of the SSD architecture and its components.

=== Backbone Networks for Feature Extraction <backbone-networks>
Explores the role of backbone networks (VGG, ResNet) in feature extraction and their impact on SSD performance.

The SSD architecture uses a base network to extract features from the input image. The base network is typically a pre-trained #acr("CNN"), such as #acr("VGG") or #acr("ResNet"), which has been trained on a large dataset like ImageNet. This assumption that features learned for other tasks can be reused for object detection is known as #acr("TL") and has proven to often be very effective in practice. @hudaEffectDiverseDataset2020, @dengInadequatelyPretrainedModels2023

==== VGG <vgg>
The #acr("VGG") network is a deep #acr("CNN") that consists of 16 or 19 layers, depending on the variant. It is known for its simplicity and effectiveness in image classification tasks. In its vanilla configuration, it takes 224x224 RGB images as input and outputs a 1000-dimensional vector of class probabilities. It only uses 3x3 convolutional layers and 2x2 max-pooling layers for feature extraction and the #acr("ReLU") activation function for non-linearity. Eventually, it employs three #acr("FC") layers for classification. The soft-max activation function is used in the final layer to predict the class probabilities. Overall, the number of trainable parameters for the #acr("VGG")-16 network is 138 million @simonyanVeryDeepConvolutional2015. #acr("VGG") is the default backbone network employed in the original #acr("SSD") paper @liuSSDSingleShot2016.

==== ResNet <resnet>
While the #acr("ResNet") architecture is mostly similar to the #acr("VGG") architecture in that it is a #acr("CNN"), it uses a couple of more advanced techniques for feature extraction.

Firstly, it uses residual blocks to address the vanishing gradient problem while significantly increasing network depth through employment of considerably more convolutional layers. Secondly, it uses batch normalization to stabilize and accelerate training. Batch normalization layers perform normalization of their input along the batch dimension, although batch in this case refers to the channels of the input tensor and not to the batch size. Batchnorm layers only have two trainable parameters, the scale and shift parameters, which are learned during training and are used to scale and shift the normalized input.

In order to find out whether #acrpl("ResNet") offer significant advantages over traditional #acrpl("CNN") models, the #acr("ResNet")-152 model, which is one of the deepest #acr("ResNet") configurations consisting of 152 convolutional layers, is used for comparison to the #acr("VGG")-16 model.

To keep the network size manageable, the #acr("ResNet")-152 model configures many of its residual blocks to use a bottleneck architecture. This architecture uses a first convolutional layer to reduce the number of channels in the input tensor, followed by additional convolutional layers that use the reduced number of channels to perform the actual feature extraction. The output of the last convolutional layer of each bottleneck is in most cases upsampled again to a higher number of channels.


=== Feature Maps and Anchor Boxes <feature-maps>
As mentioned, SSD is a single-stage detector, which means that it does not use a region proposal network to generate candidate regions for object detection. Instead, it uses a set of default boxes, also known as anchor boxes, to predict the location and class of objects in the image. The anchor boxes are generated at multiple scales and aspect ratios to cover a wide range of object sizes and shapes.

To realize this, the network will always predict a pre-defined number of bounding boxes, regardless of the number of objects in the image. Since the receptive field of the convolutional layers grow with each layer, the network will predict bounding boxes at multiple scales by using feature maps from different layers of the network.

This means that the network has several predictor heads, each of which will get a feature map from the base network as input and will predict a pre-defined number of bounding boxes for each location in the feature map. The bounding box prediction is thus defined as a regression task, in which the network generates a set of offsets relative to the default anchor boxes.

The network also predicts a vector of class probabilities for each bounding box, which indicates the confidence of the network that the bounding box contains an object of a particular class - each vector component is assigned to a class, and the sum of all components is 1. To ensure that all confidence scores lie between 0 and 1 and add up to 1, the network applies the softmax activation function to the output vector $accent(x, arrow) in bb(R)^n$ with components $x_1, dots, x_n$:
$
   accent(y, hat) = "softmax"(accent(x, arrow)) = vec(e^(x_1), dots, e^(x_n))dot 1/(sum_(i=1)^(n)e^(x_i))
$@liuSSDSingleShot2016 @bridleTrainingStochasticModel1989

=== MultiBox Loss Function <multibox-loss>
First of all, in order to compute a loss, the network needs to know which anchor boxes are positive and which are negative. To determine which ground-truth object is assigned to which anchor box, the network uses the #acr("IoU") metric, also known as the Jaccard index. That is a measure for the overlap between two bounding boxes, defined as the area of their intersection divided by the area of their union:
$
  "IoU"(B_1, B_2) = (B_1 sect B_2)/(B_1 union B_2)
$

The network then assigns each anchor box to the ground-truth object with the highest #acr("IoU") score.

* Positive Anchor Boxes: * <positive-anchor-boxes>
Positive anchor boxes are anchor boxes that have an #acr("IoU") with a ground-truth box greater than a certain threshold. In the original SSD paper, this threshold is set to 0.5.

* Negative Anchor Boxes: * <negative-anchor-boxes>
Negative anchor boxes are anchor boxes that have an #acr("IoU") whose maximum #acr("IoU") with any ground-truth box is less than the specified threshold.

Since the network is supposed to reliably detect objects in images, it needs a large quantity of anchor boxes to cover all possible object sizes and aspect ratios. For reference, the original SSD paper uses an overall 8732 anchor boxes. However, during training this leads to a large number of negative anchor boxes that do not contain any ground-truth objects.

For this reason, it is not feasible to use all negative anchor boxes for training, as this would heavily skew the loss function towards the negative class. Instead, the network uses a hard negative mining technique to select a fixed number of negative anchor boxes for training.

* Hard Negative Mining: * <hard-negative-mining>
Hard negative mining means only selecting the most difficult negative examples for training. This is done by sorting the negative anchor boxes by their confidence loss and selecting the top $k$ anchor boxes with the highest confidence loss. In practice, this is done by defining a ratio between negative and positive anchor boxes and selecting the according number of negative anchor boxes. This ensures that the network is trained on the training examples it finds most difficult to classify correctly as negative. All other negative anchor boxes are ignored, that means they do not contribute to the loss function.

Additionally, models of the SSD family employ a technique called #acr("NMS") to filter out duplicate detections and improve detection accuracy. It his highly likely that an input image of dimensions 300x300 will have multiple of the 8732 anchor boxes overlapping with each other and picking up on the same ground-truth object.

* #acr("NMS"): * <nms> After assigning each anchor box to a ground-truth object (or counting it to the negative anchor boxes) the network applies #acr("NMS") to filter out duplicate detections. Algorithmically, #acr("NMS") is carried out as follows:
+ Group all anchor boxes that have been assigned to the same ground-truth object.
+ Fort each ground-truth object, sort its assigned anchor boxes by their confidence score in descending order.
+ Select the anchor box with the highest confidence score as one of the final detections that will be accounted for in the loss functions.
+ Remove all anchor boxes that have an #acr("IoU") with the selected anchor box greater than a threshold (typically 0.5)
+ Repeat the process for the remaining anchor boxes until no more anchor boxes are left.

Once the #acr("NMS") algorithm has selected the anchor boxes relevant to the loss function, the network can compute the actual loss. That has to account for two different types of errors:
+ The first one is the localization error, which is the difference between the predicted bounding box and the ground-truth bounding box. For obvious reasons, this loss is only computed for the positive anchor boxes selected by #acr("NMS"), as it does not make sense to teach the network to regress negative bounding boxes around the background.
+ The second part is the classification error, which is the difference between the predicted class and the ground-truth class.

== Thermal Image Processing <thermal-processing>
Discusses characteristics of thermal images, preprocessing techniques (inversion, edge enhancement), and challenges specific to infrared imagery.





= Methodology <methodology>

This study employs a systematic experimental approach to evaluate the effectiveness of SSD-based neural networks for human detection in thermal imagery. The methodology encompasses dataset selection and preparation, implementation of multiple model variants with different backbone architectures, application of thermal-specific preprocessing techniques, and comprehensive evaluation metrics. The experimental design ensures reproducible results while addressing the unique challenges posed by infrared image characteristics.

*Key areas to develop:*
- Dataset description: FLIR ADAS v2, AAU-PD-T, OSU-T, M3FD, KAIST-CVPR15
- Model configurations: SSD300-VGG16 vs. SSD300-ResNet152
- Training setup: Pretrained vs. scratch initialization strategies
- Preprocessing techniques: Image inversion and edge enhancement
- Data augmentation and split strategies (train/validation/test)
- Evaluation metrics: mAP, precision, recall, inference speed
- Hardware setup and computational requirements
- Statistical significance testing approach

== Dataset Description <dataset>
Details the thermal image datasets (FLIR ADAS v2, AAU-PD-T, OSU-T, M3FD, KAIST-CVPR15) and their characteristics.

== Model Implementation <model-impl>
Explains the implementation of SSD models with different backbones and preprocessing configurations.


TODO: Incorporate switch to multi-label setup later

== Experimental Design <exp-design>
Outlines the systematic approach to comparing model variants and the evaluation framework.

= Results and Analysis <results>

The experimental evaluation reveals significant performance variations across different model configurations and preprocessing approaches when applied to thermal human detection tasks. This section presents comprehensive results from training 16 distinct model variants, combining backbone architectures (VGG16 vs. ResNet152), initialization strategies (pretrained vs. scratch), and preprocessing techniques (none, inversion, edge enhancement, combined). The analysis demonstrates clear patterns in model behavior and identifies optimal configurations for thermal surveillance applications.

*Key areas to develop:*
- Training convergence analysis: Loss curves and stability patterns
- Detection accuracy results: mAP scores across all model variants
- Preprocessing impact: Quantitative comparison of enhancement techniques
- Backbone architecture comparison: VGG16 vs. ResNet152 performance
- Initialization strategy effects: Pretrained vs. scratch training outcomes
- Computational efficiency: Inference speed and memory requirements
- Dataset-specific performance: Results breakdown by thermal dataset
- Error analysis: Common failure cases and detection limitations

== Training Performance <training-perf>
Reports training loss curves, converegence behavior, and computational requirements for different mdoel variants.


== Detection Accuracy Analysis <accuracy>
Provides detailed mAP scores and detection performance metrics for each model configuraiton and preprocessing technique.

== Preprocessing Impact Evaluation <preprocessing>
Analzyes the effects of image inversion and edge enhancement on detection performance.

= Discussion <discussion>

The experimental results provide valuable insights into the practical applicability of SSD architectures for thermal human detection systems. While certain configurations demonstrate superior performance, the choice of optimal model depends on specific deployment requirements, including accuracy thresholds, computational constraints, and operational environments. This section interprets the findings within the context of real-world surveillance applications and addresses the broader implications for thermal imaging-based security systems.

*Key areas to develop:*
- Performance trade-offs: Accuracy vs. computational efficiency analysis
- Preprocessing effectiveness: When and why certain techniques work better
- Backbone selection criteria: Situational advantages of VGG16 vs. ResNet152
- Real-world deployment implications: Edge computing considerations
- Limitations and constraints: Environmental factors affecting performance
- Comparison with existing thermal detection systems
- Cost-benefit analysis for industrial implementation
- Future optimization potential and research directions

== Model Performance Comparison <model-comparison>
Compares SSD-VGG and SSD-ResNer performance and discusses trade-offs between accuracy and computational efficiency.

== Practical Deployment Considerations <deployment>
Discusses real-world application scenarios and system requirements for thermal surveillance.

= Conclusion and Future Work <conclusion>

This thesis has systematically evaluated the application of Single Shot MultiBox Detector architectures for human detection in thermal imagery, providing empirical evidence for optimal model configurations in surveillance applications. The comprehensive analysis of 16 model variants across multiple thermal datasets has yielded practical insights for deploying neural networks in infrared-based security systems. The findings contribute to both academic understanding and industrial implementation of thermal computer vision technologies.

*Key areas to develop:*
- Key findings summary: Best-performing model configurations identified
- Methodological contributions: Systematic evaluation framework for thermal detection
- Practical implications: Guidelines for industrial thermal surveillance deployment
- Technical achievements: Successful adaptation of RGB models to thermal domain
- Research limitations: Dataset constraints and environmental factors
- Future research directions: Advanced architectures and multi-modal approaches
- Industry impact: Potential applications beyond security surveillance
- Recommendations: Implementation guidelines for practitioners

= Examples <examples>

== Figures and Tables <fig-table-examples>

Create figures or tables like this:

=== Figures <fig-example>

#figure(caption: "Image Example", image(width: 4cm, "assets/ts.svg"))

=== Tables <table-example>

#figure(
  caption: "Table Example",
  table(
    columns: (1fr, 50%, auto),
    inset: 10pt,
    align: horizon,
    table.header(
      [],
      [*Area*],
      [*Parameters*],
    ),

    text("cylinder.svg"),
    $ pi h (D^2 - d^2) / 4 $,
    [
      $h$: height \
      $D$: outer radius \
      $d$: inner radius
    ],

    text("tetrahedron.svg"), $ sqrt(2) / 12 a^3 $, [$a$: edge length],
  ),
)<table>

== Code Snippets <code-examples>

Insert code snippets like this:

#figure(
  caption: "Codeblock Example",
  sourcecode[```ts
    const ReactComponent = () => {
      return (
        <div>
          <h1>Hello World</h1>
        </div>
      );
    };

    export default ReactComponent;
    ```],
)

#pagebreak()


For example this @table references the table on the previous page.