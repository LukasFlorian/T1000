#import "@preview/supercharged-dhbw:3.4.1": *
#import "acronyms.typ": acronyms
#import "glossary.typ": glossary

#set par(spacing: 1.5em)
#show list: set block(spacing: 1.5em)
#show table.cell: set text(size: 10pt)

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
  bib-style: "ieee"
)

// Table of Contents and Content Structure

= Introduction <intro>

With the increase in security threats to critical infrastructure, automated surveillance systems have become essential for ensuring the safety and security of people, infrastructure and property at scale. The ability to detect individuals on critical infrastructure premises is crucial to preventing unauthorized access and potential damage to assets. While conventional #acr("RGB")-based surveillance systems remain prevalent in many application, they face inherent limitations in challenging scenarios such as low-light conditions, adverse weather, fog, smoke and complete darkness during nighttime @farooqObjectDetectionThermal2021.

A compelling alternative to systems operating in the visible light domain are such capturing wavelengths in the infrared spectrum and thus offering consistent detection capabilities that are fundamentally independent of ambient lighting conditions, as they rely on heat signatures emitted directly by objects. This characteristic provides unique advantages for human detection, as the human body maintains a relatively constant temperature of approximately 37°C, creating distinct thermal signatures that remain visible regardless of environmental illumination @akshathaHumanDetectionAerial2022.

The integration of deep learning architectures with thermal imaging thus opens new possibilities for automated systems that can reliably detect humans in the aforementioned scenarios with adverse conditions for conventional #acr("RGB")-based concepts. However, most state-of-the-art object detection models have been primarily developed for and trained on #acr("RGB") imagery. Given that the spectral, tectural and contrast characteristics of infrared images differ substantially from visible-light imagery, both due to the properties of those wavelengths themselves and of the sensors, those existing models might need to be adapted to achieve optimal performance.

//
This research addresses the critical need for systematic evaluation of neural object detection models specifically tailored for thermal human detection applications. The study focuses on the #acr("SSD") architecture, a prominent one-stage detection framework known for its balance between accuracy and computational efficiency. By examining multiple model variants with different backbone networks (VGG16 and ResNet152), initialization strategies (pretrained versus scratch training), and thermal-specific preprocessing techniques (image inversion and edge enhancement), this work provides comprehensive insights into optimal configurations for infrared surveillance systems.

/*
This project addresses the need for systematic evaluation of neural object detection models and preprocessing techniques tailored for thermal human detection applications. For the purpose of developing and edge-deployable network, the focus of this study lies on the #acr("SSD") architecture, a prominent one-stage detection framework with relatively low complexity compared to newer architectures like the #acr("ViT").
*/

/*
This work provides comprehensive insights into optimal configurations for infrared surveillance networks by examining multiple model variants with different:
+ *backbone networks* (#acr("VGG") and #acr("ResNet"))
+ *initialization strategies* (parameters pretrained on the #acr("RGB")-Dataset IMAGENET1K_V2 versus randomly sampled)
+ *preprocessing techniques* (inversion of the image or enhancement of its edges)
*/

The practical significance of this research extends beyond academic interest, addressing real-world challenges faced by the security and defense industry. In partnership with Airbus Defence & Space, this project explores the development of cost-efficient, edge-deployable thermal surveillance solutions that can operate reliably in challenging environments where traditional #acr("RGB") systems fail.

== Research Objectives and Contributions

This thesis makes several key contributions to the field of thermal image processing and computer vision:

*TODO: Reconsider Objectives vs. Contributions, place latter in the conclusion*

+ *Preprocessing Technique Analysis*: Quantitative evaluation of thermal-specific image enhancement methods, including polarity inversion and edge enhancement, and their impact on detection accuracy.

+ *Backbone Network Comparison*: Detailed comparison between VGG16 and ResNet152 architectures in the context of thermal imagery, addressing the trade-offs between model complexity and performance.

+ *Practical Implementation Guidelines*: Development of actionable recommendations for deploying thermal surveillance systems in real-world environments, considering computational constraints and accuracy requirements.

+ *Dataset Integration Framework*: Unified evaluation approach across five diverse thermal datasets (FLIR ADAS v2 @FREEFLIRThermal, AAU-PD-T @hudaEffectDiverseDataset2020, OSU-T @davisTwoStageTemplateApproach2005, M3FD @liuTargetawareDualAdversarial2022, KAIST-CVPR15 @hwangMultispectralPedestrianDetection2015), enabling robust performance assessment.

== Thesis Structure

The remainder of this thesis is structured to provide a comprehensive examination of thermal human detection using neural networks. @literature presents a thorough review of object detection fundamentals, SSD architecture principles, and thermal image processing techniques, establishing the theoretical foundation for the experimental work. @methodology details the systematic approach employed for model evaluation, including dataset preparation, experimental design, and evaluation metrics. @results presents comprehensive performance analysis across all model configurations and preprocessing techniques. @discussion interprets the findings within the context of practical deployment scenarios and industrial requirements. Finally, @conclusion synthesizes the key contributions and outlines directions for future research in thermal surveillance technologies.


/*
This research represents a significant step toward practical implementation of AI-powered thermal surveillance systems, providing the empirical foundation necessary for informed decision-making in security-critical applications where reliable human detection is paramount.
*/

= Literature Review and Theoretical Background <literature>

The field of object detection has undergone significant evolution from traditional computer vision techniques to sophisticated deep learning architectures. Understanding this progression is essential for contextualizing the current work's contribution to thermal image analysis. This section examines the theoretical foundations of object detection, with particular emphasis on the Single Shot MultiBox Detector (SSD) architecture and its applicability to thermal imagery processing challenges.

== Object Detection Fundamentals <obj-detection>
Most object detection methods can be broadly categorized into two main approaches: traditional methods and deep learning-based methods. The former mainly rely on handcrafted features and sliding window techniques @violaRapidObjectDetection2001a, while newer approaches in this field leverage deep #acrpl("CNN") or #acr("ViT") architectures to automatically learn features from data @dosovitskiyImageWorth16x162021 @alqahtaniBenchmarkingDeepLearning2024.

=== Traditional Object Detection Methods <traditional-methods>
Simple approaches to object detection entail applying manually constructed feature detector kernels in a sliding window fashion to images. 

One example of this is the *Viola-Jones-Algorithm* @violaRapidObjectDetection2001a:
+ The algorithm first computes the integral image of the input—a representation where each pixel stores the cumulative sum of intensities from the top-left corner to its position. This allows constant-time calculation of the sum of pixel values within any rectangular region, using only four array references (the corners of the rectangle).

+ It then applies Haar-like features - simple rectangular patterns (e.g., edge, line, or center-surround detectors) - to rapidly identify potential regions of interest. Each feature’s value is derived by subtracting the sum of pixels in one rectangle from the sum in an adjacent rectangle, leveraging the integral image for efficiency.

+ A strong classifier is constructed by training a series of weak classifiers (typically decision stumps) using #acr("AdaBoost"). These weak classifiers focus on individual Haar-like features, while the cascaded structure enables early rejection of non-object windows, significantly reducing computation.

+ Finally, the algorithm scans the image using a sliding window, classifying each subwindow with the cascaded classifier. Windows that pass all stages of the cascade are marked as containing the target object.

Other approaches employ #acr("HOG") descriptors. The #acrpl("HOG") are attained by dividing the image into a grid of cells, contrast-normalizing them and then computing the vertical as well as horizontal gradients of their pixels @dalalHistogramsOrientedGradients2005. The gradients for each cell are accumulated in a one-dimensional histogram which serves as that cell's feature vector @dalalHistogramsOrientedGradients2005. After labeling the cells in the training data, a #acr("SVM") can be trained to find an optimal hyperplane separating the feature vectors corresponding to the object that should be detected from those that do not contain the object @cortesSupportvectorNetworks1995.

=== Deep Learning-Based Object Detection <deep-learning-detection>
However, those methods are either highly dependent on engineering the correct priors, such as the Haar-like features, or limited to binary classification scenarios, as is the case for #acr("HOG")-based #acrpl("SVM") @cortesSupportvectorNetworks1995. Thus, newer Object Detection methods employ more complex deep-learning architectures that require less manual feature engineering. The best-performing models nowadays are #acrpl("ViT") using Attention mechanisms @dosovitskiyImageWorth16x162021 to learn relationships between patterns in different parts of images. However, they will not be further examined in this thesis, due to computational constraints that make them unfeasible for the edge-deployable solution sought in this work @dosovitskiyImageWorth16x162021.

Relevant for this examination are their predecessors, #acrpl("CNN"). The main mechanism they use to extract information from images are convolutional layers. Those convolutional layers get passed an image in the form of a tensor and perform matrix multiplication on that input tensor and a kernel tensor in a sliding window fashion to compute subsequent feature maps. Those will be passed on as input to the next layer. @lecunHandwrittenDigitRecognition1989

At their core, these convolutional layers do not work inherently different from #acr("FC") layers that compute several weighted sums across all components of the input tensor. More specifically, fully connected layers can be described as convolutional layers whose kernel dimensions are identical to those of the input tensor.

Resorting to smaller kernels, however, serves as a prior making use of the heuristic that in most cases, the features composing an object in an image lie closely together. Thus, it is not necessary to process the entire image to detect an object that occupies only part of it. Convolutional neural nets hence save computational resources by focusing on smaller regions. In many cases it is advantageous to use those savings to increase network depth in order to make it possible for the network to learn more complex high-level features in subsequent layers.

Object detection, as opposed to image classification, consists of two main tasks; locating where an object is and classifying which class it belongs to. In the context of machine learning, that means  regression must be used to approximate the location of an object and the concept of classification is applied to determine its class. #acrpl("CNN") solving these tasks can be categorized into two main categories:

*Two-Stage Detectors* split the detection objective into the two tasks mentioned before. The first stage proposes regions of interest and the second stage classifies which object they contain. In more detail, that means regressing bounding boxes and assessing the "objectness" of that region, for example by using logistic regression. If the confidence this region contains an object exceeds a given threshold, the second stage then classifies the object in that region. That requires a second pass of the extracted region through a classifier network. This two-stage approach can be computationally expensive, especially when dealing with a large number of proposals. Examples of two-stage detectors include #acrpl("R-CNN") @girshickRichFeatureHierarchies2014, Fast #acrpl("R-CNN") @girshickFastRCNN2015, and Faster #acrpl("R-CNN") @renFasterRCNNRealTime2016.

*Single-Stage Detectors*, on the other hand, perform both tasks simultaneously in a single pass through the network. That means passing the image through a network that both regresses bounding boxes and classifies objects in those boxes at the same time. Examples include #acr("YOLO") @redmonYouOnlyLook2016  and #acr("SSD") @liuSSDSingleShot2016. This approach can be faster but may sacrifice some accuracy compared to two-stage detectors, as the feature extractor is not optimized for both tasks.

Given the computational constraints imposed by the requirement for edge-deployment, single-stage detectors were chosen. Past research has shown that #acr("SSD")-variants with Inception-v2 and MobileNet-v1 backbones perform notably faster than their Faster #acr("R-CNN") counterparts, namely 4 to 7 times as fast @akshathaHumanDetectionAerial2022.

Furthermore, benchmarks of #acr("SSD") and #acr("YOLO") on the MS COCO dataset yielded similar results favoring #acr("SSD") in terms of speed when deployed on edge devices, namely the Raspberry Pi 4 both with and without a dedicated #acr("TPU") @alqahtaniBenchmarkingDeepLearning2024. #acr("YOLO") did deliver higher #acr("mAP") scores, but the difference was not significant enough to justify the trade-off in speed, in particular taking into account the benefit of speed for real-time applications when multiple images are captured each second and fast enough processing allows for multiple attempts at detection. Additionally, the #acr("SSD") models tested consumed less energy than their #acr("YOLO") counterparts @alqahtaniBenchmarkingDeepLearning2024, making them a more suitable choice that minimizes the need for human intervention to replace the battery of the edge device, which is a significant factor in the cost of deployment and maintenance of the system.

== #acr("SGD") as Optimizer in Deep Learning <sgd-optimizer>
Deep Learning Models are optimized by minimizing the loss function $L(accent(y, hat), y)$, which is a measure of the difference between the predicted output $accent(y, hat)$ and the expected output $y$, i.e. the ground truth @erhanScalableObjectDetection2013. The loss function is typically a differentiable function, which means that it can be used to compute the gradient of the loss with respect to the model parameters. The gradient is then used to update the model parameters in the direction that minimizes the loss function, hence the name gradient descent @ruderOverviewGradientDescent2017.

This happens in reverse order of the forward pass, since gradients of the loss function relative to earlier layer's parameters are computed using the chain rule, which is why this process is also called backpropagation. More specifically, the algorithm performs the following steps:

+ It first computes the gradient of the loss function with regard to a layer's output.
+ Using the chain rule, it computes the gradient of the loss function with respect to the layer's parameters.
+ Based on the results, it updates the layer parameters in the opposite direction of the gradient
+ Now the algorithm moves on to the next (earlier) layer and repeats the process until it reaches the first layer of the network.

The challenges posed by this process are discussed in @vgp


In most cases, the training dataset is too large to compute the gradient with respect to all data at once, which is called #acr("BGD"). Instead, the optimization takes place in so-called mini-batches of training data of a fixed size, under the assumption that the gradient computed relative to a mini-batch of training data is a good approximation of the gradient that would be obtained if the calculation was performed across the entire training dataset. @ruderOverviewGradientDescent2017

This differentiation of the loss function with regard to a mini-batch of training data is called #acr("MBGD"), but it is also commonly referred to as #acr("SGD") in the literature and will be called that for simplicity.

As desribed before, #acr("SGD") is a first-order optimization algorithm, which means that it only considers the first-order derivatives of the loss function with respect to the model parameters. This is in contrast to second-order optimization algorithms, such as Newton's method, which consider the second-order derivatives of the loss function as well. However, first-order optimization algorithms are generally preferred in deep learning because they are computationally more efficient and can be easily implemented on hardware accelerators such as #acrpl("GPU") and #acrpl("TPU"), which is why second-order optimization algorithms will not be further discussed in this work. @ruderOverviewGradientDescent2017

=== The #acrf("VGP") <vgp>

One limitation of #acr("SGD") lies in the so-called #acrf("VGP"), which occurs due to the calculation of the partial derivates by means of the chain rule. The chain rule for differentiation states that the derivative of a composite function is the product of the derivatives of its components, as shown in @chain-rule.

$ 
(f(g(x)))' = f'(g(x)) dot g'(x)
$<chain-rule>

In this equation, $f$ and $g$ are the functions applying the linear transformations corresponding to a convolutional layer, its successor layer and their activation functions, respectively. Since both convolutional kernels contain a large number of parameters, each individual parameter only has a small impact on the overall loss function. As a result, the product of the derivatives of the loss function with respect to the parameters of the two layers can become very small, leading to a phenomenon known as the #acr("VGP").

In the context of deep learning, this means that deeper models are particularly likely to suffer from this problem, as more layers amplify the core issue. One approach to mitigate this problem is to introduce the so-called residual layers allowing for an identity mapping of the input to the output of a layer, which is a key component of the #acr("ResNet") architecture.

=== Residual Layers to Mitigate the #acr("VGP") <resnet-arch>

Residual layers are a key component of the #acr("ResNet") as well as other modern architectures, and are designed to mitigate the #acr("VGP") problem. The idea behind residual layers is to introduce a shortcut connection that allows the input to the layer to be added to the output of the layer. This shortcut connection allows the gradient to "flow directly through the layer", which helps to prevent the gradient from vanishing as it is backpropagated through the network. @residual-fn shows the residual layer, where $F$ is the residual function, $x$ is the input to the layer, and $accent(y, hat)$ is the output of the layer. The residual function $F$ is typically a stack of convolutional layers, #acr("BN") layers, and #acr("ReLU") activation functions. @heDeepResidualLearning2015
$
  accent(y, hat) = F(x) + x
$<residual-fn>

#figure(
  image("assets/res_layer.png", width: 40%),
  caption: [Residual layer composed of layers with functions $f$ and $g$ with a shortcut connection, inspired by @heDeepResidualLearning2015],
  gap: 10pt,
  placement: top
)<residual-graphic>

If the function $F$ is a composite function $F(x) = f(g(x))$, then @residual-fn can be transformed as seen in @residual-derivative to attain the derivative of $accent(y, hat)$ with respect to $x$, which given by the chain rule from @chain-rule.
$
  accent(y, hat)'(x) &= F'(x) + x'\
  &= f'(g(x)) dot g'(x) + 1
$<residual-derivative>
Here, the derivative of $x$ with respect to itself is 1. In reality, $x$ would be the result of a previous layer's function $h(x_0)$, which implies for the derivative of $accent(y, hat)$ with respect to $x_0$ the relationship introduced in @derivative-chain.
$
  accent(y, hat)(x)' &= accent(y, hat)(h(x_0))'\
  &= accent(y, hat)'(h(x_0)) dot h'(x_0)\
  &= (F'(h(x_0)) + 1) dot h'(x_0)\
  &= f'(g(h(x_0))) dot g'(h(x_0)) dot h'(x_0) + h'(x_0)
$<derivative-chain>
It becomes apparent that introducing residual layers and identity shortcuts helps to mitigate the #acr("VGP") by preserving the gradient with respect to previous layers' outputs without multiplying them with many small derivatives of subsequent layers. $F(x)$ can be an arbitrarily deep function and the gradient with respect to $x_0$ will still be $h'(x_0)$ plus a term that is multiplied by the derivatives of the residual function $F(x)$. This is in contrast to the gradient in a standard #acr("CNN") where repeated application of the chain rule is likely to cause the gradient to vanish.

When the input and output dimensions of a residual layer do not match, a linear projection is used to match the dimensions of the input and output before adding the two together. This linear projection is typically implemented using a 1x1 convolutional layer @heDeepResidualLearning2015. Even though this practice introduces additional parameters that need to be learned and makes the identity shortcut actually not an identity function anymore, it is still beneficial to the training process, as the residual function can be used to significantly increase the network's depth while preserving the gradient flow.

*TODO ONCE TEXT IS DONE: Residual Layer visualization*

== Single Shot MultiBox Detector (SSD) Architecture <ssd-arch>
The SSD architecture is a single-stage detector that uses a base network to extract features from the input image and then applies additional convolutional layers as well as #acr("FC") layers to predict bounding boxes and class scores for each feature map. The following sections provide a more detailed explanation of the SSD architecture and its components.

=== Backbone Networks for Feature Extraction <backbone-networks>
Explores the role of backbone networks (VGG, ResNet) in feature extraction and their impact on SSD performance.

The SSD architecture uses a base network to extract features from the input image. The base network is typically a pre-trained #acr("CNN"), such as #acr("VGG") or #acr("ResNet"), which has been trained on a large dataset like ImageNet. This assumption that features learned for other tasks can be reused for object detection is known as #acr("TL") and has proven to often be very effective in practice. @hudaEffectDiverseDataset2020, @dengInadequatelyPretrainedModels2023

*The #acrf("VGG") network* is a deep #acr("CNN") that consists of 16 or 19 layers, depending on the variant. It is known for its simplicity and effectiveness in image classification tasks. In its vanilla configuration, it takes 224x224 #acr("RGB") images as input and outputs a 1000-dimensional vector of class probabilities. It only uses 3x3 convolutional layers and 2x2 max-pooling layers for feature extraction and the #acr("ReLU") activation function for non-linearity. Eventually, it employs three #acr("FC") layers for classification. The soft-max activation function is used in the final layer to predict the class probabilities. Overall, the number of trainable parameters for the #acr("VGG")-16 network is 138 million @simonyanVeryDeepConvolutional2015. #acr("VGG") is the default backbone network employed in the original #acr("SSD") paper @liuSSDSingleShot2016.


*While the #acrf("ResNet")* architecture is mostly similar to the #acr("VGG") architecture in that it is a #acr("CNN"), it uses a couple of more advanced techniques for feature extraction.

Firstly, it utilizes residual blocks to address the #acr("VGP") while significantly increasing network depth through employment of considerably more convolutional layers @heDeepResidualLearning2015. Secondly, it uses #acr("BN") to stabilize and accelerate training. #acr("BN") layers perform normalization of their input along the batch dimension, although batch in this case refers to the channels of the input tensor and not to the batch size. #acr("BN") layers only have two trainable parameters, the scale and shift parameters, which are learned during training and are used to scale and shift the normalized input. The stabilizing effect stems from the fact that #acr("BN") reduces the covariate shift between layers during update steps. The covariate shift describes the change in the distribution of the input to a layer due to the change in the parameters of the previous layer, which significantly slows down training progress when using #acr("SGD"). @ioffeBatchNormalizationAccelerating2015

Secondly, a bottleneck architecture ensures that deeper networks exploiting the solution offered by residual layers do not become too large in overall parameter count. This architecture uses an initial convolutional layer to reduce the number of channels in the input tensor, followed by additional convolutional layers that use the reduced number of channels to perform the actual feature extraction. The output of the last convolutional layer of each bottleneck is in most cases upsampled again to a higher number of channels. @heDeepResidualLearning2015

In order to find out whether #acrpl("ResNet") offer significant advantages over traditional #acrpl("CNN") models, the #acr("ResNet")-152 model, which is one of the deepest #acr("ResNet") configurations consisting of 152 convolutional layers, is used for comparison to the #acr("VGG")-16 model.


=== Feature Maps and Anchor Boxes <feature-maps>
As mentioned, SSD is a single-stage detector, which means that it does not use a region proposal network to generate candidate regions for object detection. Instead, it utilizes a set of default boxes, also known as anchor boxes, to predict the location and class of objects in the image. The anchor boxes are generated at multiple scales and aspect ratios to cover a wide range of object sizes and shapes.

To realize this, the network will always predict a pre-defined number of bounding boxes, regardless of the number of objects in the image. Since the receptive field of the convolutional layers grow with each layer, the network will predict bounding boxes at multiple scales by using feature maps from different layers of the network.

This means that the network has several predictor heads, each of which will get a feature map from the base network as input and will predict a pre-defined number of bounding boxes for each location in the feature map. The bounding box prediction is thus defined as a regression task. Let $accent(d_i, arrow)$ be the $i$-th default anchor box, encoded by its center coordinates $d^("cx"), d^("cy")$, its width $d^w$, and its height $d^h$. The network will predict an offset vector $accent(l_i, arrow)$ relative to each default anchor box $accent(d_i, arrow)$ such that the final predicted bounding box $accent(b_i, arrow)$ can be calculated using @bbox-regression @liuSSDSingleShot2016.

$
  accent(b_i, arrow) &= vec(
    b_i^"cx",
    b_i^"cy",
    b_i^w,
    b_i^h
  )
  &= vec(
    d_i^"cx" + l_i^"cx" dot d_i^w,
    d_i^"cy" + l_i^"cy" dot d_i^h,
    d_i^w dot e^(l_i^w),
    d_i^h dot e^(l_i^h)
  )
$<bbox-regression>

The network also predicts a vector of class probabilities for each bounding box, which indicates the confidence of the network that the bounding box contains an object of a particular class - each vector component is assigned to a class, and the sum of all components is 1. To ensure that all confidence scores lie between 0 and 1 and add up to 1, the network applies the softmax activation function from @softmax-eqn to the output vector $accent(p, arrow) in bb(R)^C$ with components $p_1, dots, p_C$ @liuSSDSingleShot2016.
$
   accent(p, hat) = "softmax"(accent(x, arrow)) = vec(e^(p_1), dots, e^(p_C))dot 1/(sum_(i=1)^(C)e^(p_i))
$<softmax-eqn>@bridleTrainingStochasticModel1989

=== MultiBox Loss Function <multibox-loss>
First of all, in order to compute a loss, the network needs to know which anchor boxes contain an object and which do not. To determine which ground-truth object is assigned to which anchor box, the network uses the #acr("IoU") metric @liuSSDSingleShot2016, also known as the Jaccard index @costaFurtherGeneralizationsJaccard2021. That is a measure for the overlap between two bounding boxes $B_1$ and $B_2$, defined as the area of their intersection divided by the area of their union, as shown in @iou-eqn @costaFurtherGeneralizationsJaccard2021.
$
  "IoU"(B_1, B_2) = (|B_1 inter B_2|)/(|B_1 union B_2|)
$<iou-eqn>

This #acr("IoU") metric is then used to match the regressed boxes to the ground-truth boxes. It is calculated for each pair of anchor boxes and ground-truth boxes. Based on the #acr("IoU") scores, the network determines which anchor boxes are positive and which are negative. Negative anchor boxes are those that do not have an #acr("IoU") above a certain threshold with any ground-truth box. Positive boxes, on the other hand, satisfy the condition that they must have a certain minimum overlap with any ground-truth box. If they overlap sufficiently with multiple ground-truth boxes, the one with the highest #acr("IoU") is assigned to that object. @liuSSDSingleShot2016, @erhanScalableObjectDetection2013

The result is an unambiguous mapping from positive anchor boxes to ground-truth boxes.

Since the network is supposed to reliably detect objects in images, it needs a large quantity of anchor boxes to cover all possible object sizes and aspect ratios. For reference, the paper introducing the #acr("SSD") uses an overall 8732 anchor boxes @liuSSDSingleShot2016. However, during training this leads to a large number of negative anchor boxes that do not contain any ground-truth objects.

For this reason, it is not feasible to use all negative anchor boxes for training, as this would heavily skew the loss function towards the negative class. Instead, the network uses a #acr("HNM") technique to select a fixed number of negative anchor boxes for training.

*#acrf("HNM")* means only selecting the most difficult negative examples for training. This is done by sorting the negative anchor boxes by their confidence loss and selecting the top $k$ anchor boxes with the highest confidence loss. In practice, this is done by defining a ratio between negative and positive anchor boxes and selecting the according number of negative anchor boxes. This ensures that the network is trained on the training examples it finds most difficult to classify correctly as negative. All other negative anchor boxes are ignored, that means they do not contribute to the loss function. @liuSSDSingleShot2016

Additionally, models of the SSD family employ a technique called #acr("NMS") to filter out duplicate detections and improve detection accuracy. It his highly likely that an input image of dimensions 300x300 will have multiple of the 8732 anchor boxes overlapping with each other and picking up on the same ground-truth object.

*#acrf("NMS")* is applied after assigning each anchor box to a ground-truth object (or counting it to the negative anchor boxes) to filter out duplicate detections. Algorithmically, #acr("NMS") is carried out as follows @erhanScalableObjectDetection2013 @liuSSDSingleShot2016:
+ The algorithm begins by grouping all anchor boxes that have been assigned to the same ground-truth object.
+ For each ground-truth object, the assigned anchor boxes are sorted by their confidence score in descending order.
+ The anchor box with the highest confidence score is selected as one of the final detections that will be accounted for in the loss function.
+ All anchor boxes that exhibit an #acr("IoU") greater than a threshold (typically 0.5) with the selected anchor box are removed from consideration.
+ This process is repeated for the remaining anchor boxes until no more anchor boxes are left.

Once the #acr("NMS") algorithm has selected the anchor boxes relevant to the loss function, the network can compute the actual loss. That has to account for two different types of errors, localization and classification error @liuSSDSingleShot2016 @erhanScalableObjectDetection2013.

The former is the difference between the predicted bounding box and the ground-truth bounding box @erhanScalableObjectDetection2013. For obvious reasons, this loss is only computed for the positive anchor boxes selected by #acr("NMS"), as it does not make sense to teach the network to regress negative bounding boxes around the background @liuSSDSingleShot2016. This localization error uses the Smooth L1 loss function from @l1-eqn, which is often used in object detection tasks as it combines the characteristics of other alternatives like the L1 and L2 loss to allow for a robust training process @elharroussLossFunctionsDeep2025.

$
  "SmoothL1"(y, accent(y, hat)) = cases(
    1/2(y - accent(y, hat))^2 &"   if" |y - accent(y, hat)| < 1,
    |y - accent(y, hat)| - 1/2 &"   otherwise"
  )
$<l1-eqn>


Plugging in the ground-truth bounding box vector $accent(g_j, arrow)$ for the bounding box vector derived through @bbox-regression allows to formulate the conversion from $accent(g_j, arrow)$ to the offset vector $accent(g_j, hat)$ the network is expected to predict respective to any corresponding default anchor box $accent(d_i, arrow)$:

$
  accent(g_i, arrow) &= vec(
    g_j^"cx",
    g_j^"cy",
    g_j^"w",
    g_j^"h"
  )
  &= vec(
    d_i^"cx" + accent(g, hat)_j^"cx" dot d_i^w,
    d_i^"cy" + accent(g, hat)_j^"cy" dot d_i^h,
    d_i^w dot e^(accent(g, hat)_j^w),
    d_i^h dot e^(accent(g, hat)_j^h)
  )
  --> accent(g_j, hat) &= vec(
    g_j^"cx",
    g_j^"cy",
    ln(g_j^"w"/d_i^"w"),
    ln(g_j^"h"/d_i^"h")
  )
$<offset-conversion>

This representation can be used to determine the $"SmotthL1"$ localization loss. Let $"P"$ be the set of positive anchor indices selected by #acr("NMS"), $"G"$ be the set of ground-truth box indices and $a_"ij"$ the indicator that the $i$-th ground-truth box is assigned to the $j$-th anchor box. Then the localization loss $L_"loc"$ is defined as in @loc-loss @liuSSDSingleShot2016.

$
  L_("loc")(a, l, g, "P", "G") &= sum_(i in "P")(sum_(j in "G")(sum_(m in {"cx", "cy", w, h}) a_"ij" "SmoothL1"(accent(g, hat)_j^m, l_i^m)))
$<loc-loss>

The second component of the overall loss function is the classification error, which is the difference between the predicted class and the ground-truth class. This is computed for the hard negative as well as positive anchor boxes. The classification loss for any given anchor box with confidence score vector $accent(p, arrow) in (0,1)^C$, one-hot ground-truth class vector $accent(y, arrow) in {0,1}^C$ and $C$ classes to detect is computed using the cross-entropy loss from @ce-loss. The cross-entropy loss is specifically designed for multi-class classification problems. @elharroussLossFunctionsDeep2025
$
  "CrossEntropy"(accent(y, arrow), accent(p, arrow)) = -sum_(c=1)^C y_c dot log(p_c)
$<ce-loss>

Where $C$ is the number of classes to detect, $y_(i,c) in {0,1}$ is the binary ground-truth label whether sample $i$ belongs to class $c$, and $p_(i,c) in (0,1) $ is the predicted probability that $c$ is the correct class for $i$.

After adjusting for the number of positive anchor boxes - such that training images containing more positive anchor boxes do not contriubte overproportionally to the loss - @class-loss describes the classification loss, where $"N"$ is the set of negative anchor boxes selected by #acr("HNM").
$
  L_("class")("P", "N", y, p) = sum_(i in "P") "CrossEntropy"(accent(y_i, arrow), accent(p_i, arrow)) + sum_(i in "N") "CrossEntropy"(accent(y_i, arrow), accent(p_i, arrow))
$<class-loss>@liuSSDSingleShot2016

The final loss is a weighted sum of the localization and confidence loss with a scaling factor $alpha$, adjusted for the number of positive anchor boxes after #acr("NMS").
$
  L(a, y, p, l, g, "P", "N", "G") = 1/(|"P"|) dot (L_("Loc")(a, l, g, "P", "G") + alpha dot L_("class")("P", "N", y, p))
$<total-loss>@liuSSDSingleShot2016

Propagating that loss from @total-loss backwards through the network, the weights of the network are updated using #acr("SGD") with a learning rate $eta$, which means that each parameter's specific gradient it multiplied by $eta$ before performing the parameter update step. The basic idea is that since the result of @total-loss is a measure of the network's performance, adjusting the network parameters such that the loss function approaches a local or even global minimum should yield a better performing model also by human standards - that encapsulates better fitting bounding boxes as well as more accurate classification of the objects in the images.

== Thermal Image Processing <thermal-processing>

Thermal imaging presents unique challenges and opportunities for computer vision applications compared to conventional #acr("RGB") imagery. Understanding these characteristics and developing appropriate preprocessing techniques is crucial for optimizing object detection performance in infrared surveillance systems.

=== Characteristics of Thermal Images <thermal-characteristics>

Thermal images fundamentally differ from visible-light imagery in several key aspects that directly impact neural network performance. Unlike #acr("RGB") images that capture reflected light, thermal cameras detect electromagnetic radiation in the infrared spectrum (typically 8-14 μm), creating images based on the heat signatures emitted by objects @farooqObjectDetectionThermal2021.

The most distinctive characteristic of thermal imagery is its independence from ambient lighting conditions. Since thermal cameras detect heat radiation rather than reflected light, they provide consistent imaging capabilities in complete darkness, fog, smoke, and other challenging environmental conditions where traditional #acr("RGB") systems fail @akshathaHumanDetectionAerial2022. This makes thermal imaging particularly valuable for surveillance applications.

However, thermal images also present unique challenges for object detection models originally designed for visible-light imagery. The spectral characteristics result in different texture patterns, contrast relationships, and edge definitions compared to #acr("RGB") images. Additionally, thermal sensors often produce images with lower spatial resolution and different noise characteristics, requiring specialized preprocessing approaches to optimize detection performance. @beyererCNNbasedThermalInfrared2018

=== Preprocessing Techniques for Thermal Detection <thermal-preprocessing>

To address the unique characteristics of thermal imagery and improve object detection accuracy, this study implements three primary preprocessing techniques: normalization, polarity inversion, and edge enhancement. These techniques are designed to adapt #acr("RGB")-trained models to thermal domain characteristics while preserving essential spatial and thermal information.

*Normalization* serves as the fundamental preprocessing step, ensuring consistent input scaling across all thermal images and enabling proper transfer of learned features from the #acr("RGB") domain for the pretrained model backbones. Following standard practice, input images are normalized using the ImageNet dataset statistics with channel-wise means of [0.485, 0.456, 0.406] and standard deviations of [0.229, 0.224, 0.225] @dengImageNetLargeScaleHierarchical.

The normalization process transforms pixel intensities according to @norm-eqn, where $I_"norm"$ represents the normalized image, $I_"raw"$ is the input thermal image, $mu$ is the mean, and $sigma$ is the standard deviation for each channel.

$
  I_"norm" = (I_"raw" - mu) / sigma
$<norm-eqn>

*Polarity inversion* addresses the fundamental difference in thermal image representation compared to natural images. In mopst thermal imaging scenarios, humans appear as bright objects against darker backgrounds due to their higher body temperature. However, thermal scene characteristics such as high ambient temperatures can result in inverted polarity where humans appear dark against bright backgrounds @hudaEffectDiverseDataset2020.

Since thermal images might exhibit either polarity, inverting one of the channels ensures that at least one channel maintains consistent human-background contrast relationships across all images. This approach is inspired by the work of @hudaEffectDiverseDataset2020, which demonstrated the effectiveness of various preprocessing techniques in improving transfer learning performance across diverse thermal datasets. Addtionally, @hudaEffectDiverseDataset2020 also states that specifically polarity-inverted thermal images have a close resemblance to grayscale-converted #acr("RGB") images.

Since most #acr("CNN") architectures and pretrained weights are optimized for detecting darker objects (edges, shadows) against lighter backgrounds in #acr("RGB") imagery, thermal images with inverted polarity may not align with these learned features @hudaEffectDiverseDataset2020. Polarity inversion preprocessing ensures consistent object-background contrast relationships by applying a simple pixel-wise transformation according to @inversion-eqn.

$
  I_"inverted" = 1.0 - I_"original"
$<inversion-eqn>

Here, the constant $1.0$ is chosen simply to ensure quick computation of the inverted channel and because it lies close to the maximum pixel intensity after normalization.

*Edge enhancement* preprocessing aims to strengthen the boundary definition between objects and backgrounds in thermal images, which often exhibit softer transitions compared to #acr("RGB") imagery due to heat diffusion effects or wind-induced heat dissipation @hudaEffectDiverseDataset2020. The implementation combines Gaussian blur preprocessing to reduce noise and smooth the image, followed by Sobel edge detection to create enhanced edge representations.

The specific kernel used for blurring is defined in @gaussian-kernel and is chosen such that it is normalized to sum to $1.0$ and preserves spatial characteristics while decreasing intra-channel variance.

$
  K_"Gaussian" = mat(
    1/16, 1/8, 1/16;
    1/8, 1/4, 1/8;
    1/16, 1/8, 1/16;
  )
$<gaussian-kernel>

Subsequently, Sobel operators are applied to detect horizontal and vertical edges. The Sobel kernels $S_x$ and $S_y$ for horizontal and vertical edge detection are defined as @burnhamComparisonRobertsSobel:

$
  S_x = mat(
    -1, 0, 1;
    -2, 0, 2;
    -1, 0, 1
  ), quad S_y = mat(
    -1, -2, -1;
     0,  0,  0;
     1,  2,  1
  )
$<sobel-kernels>

The final edge magnitude is computed by combining both directional gradients according to @edge-magnitude, providing enhanced boundary information that emphasizes object contours in thermal imagery.

$
  E = sqrt((I * S_x)^2 + (I * S_y)^2)
$<edge-magnitude>@burnhamComparisonRobertsSobel

Where $*$ denotes the convolution operation. This edge-enhanced representation provides additional geometric information that complements the thermal intensity data, potentially improving the model's ability to localize and classify human subjects in infrared images.

=== Preprocessing Combination Strategies <combination-strategies>

The preprocessing techniques can be applied individually or in combination to optimize detection performance for specific thermal imaging scenarios. The normalization is always applied in order to ensure consistent transfer learning applicability. Thus, the experimental design evaluates four distinct preprocessing configurations:

1. *Normalization only*: Baseline preprocessing maintaining original thermal characteristics and maximizing transfer learning performance
2. *Normalization + Inversion*: Addressing polarity variations in thermal scenes
3. *Normalization + Edge Enhancement*: Emphasizing geometric features for improved localization
4. *Normalization + Inversion + Edge Enhancement*: Combined approach leveraging both polarity correction and geometric enhancement

= Methodology <methodology>

This study employs a systematic experimental approach to evaluate the effectiveness of SSD-based neural networks for human detection in thermal imagery. The methodology encompasses dataset selection and preparation, implementation of multiple model variants with different backbone architectures, application of thermal-specific preprocessing techniques, and comprehensive evaluation metrics. The experimental design ensures reproducible results while addressing the unique challenges posed by infrared image characteristics.

*Key areas to develop:*
- Dataset description: FLIR ADAS v2, AAU-PD-T, OSU-T, M3FD, KAIST-CVPR15
- Training setup: Pretrained vs. scratch initialization strategies
- Preprocessing techniques: Image inversion and edge enhancement
- Data augmentation and split strategies (train/validation/test)
- Evaluation metrics: mAP, precision, recall, inference speed
- Hardware setup and computational requirements
- Statistical significance testing approach

== Dataset Description <dataset>

The experimental evaluation employs five complementary thermal datasets that collectively provide comprehensive coverage of diverse infrared imaging scenarios and human detection challenges. The strategic selection of these datasets addresses the fundamental objective of creating a robust, generalizable detection system capable of operating across varying environmental conditions, camera configurations, and thermal imaging scenarios that are representative of real-world surveillance applications.

#figure(
  caption: [Comprehensive thermal dataset specifications and characteristics],
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto),
    inset: 6pt,
    align: horizon,
    table.header(
      [*Dataset*],
      [*Resolution*],
      [*Images*],
      [*Data Splits*],
      [*Environment*],
      [*Camera Setup*],
      [*Annotated Classes*],
      [*Objects*],
    ),

    [FLIR ADAS v2], 
    [640×512], 
    [10,495], 
    [Train/Val/Test],
    [Automotive roads, urban/highway, day/night, adverse weather],
    [Vehicle-mounted, forward-facing],
    [Person, car, bicycle, other vehicle, animal],
    [~98,000],

    [AAU-PD-T], 
    [640×480], 
    [2,941],
    [Train: 1,941\nTest: 1,000],
    [Controlled outdoor sports fields, winter conditions],
    [Elevated stationary (9m height)],
    [Person],
    [7,809],

    [OSU-T], 
    [320×240], 
    [17,373],
    [Test only],
    [University campus, natural outdoor pedestrian areas],
    [Elevated stationary (building rooftop, 3 stories)],
    [Person],
    [~15,000],

    [M3FD Detection], 
    [640×512], 
    [4,200],
    [Train: 3,780\nTest: 420],
    [Urban driving, challenging visibility conditions],
    [Vehicle-mounted dual-modal],
    [Person, vehicle, bicycle, traffic objects],
    [~25,000],

    [KAIST-CVPR15], 
    [640×512], 
    [95,328],
    [Train/Val subsets],
    [Urban pedestrian scenarios, day/night cycles],
    [Vehicle roof-mounted, ego-centric],
    [Person, people, cyclist],
    [103,128],
  ),
)<dataset-table>

The dataset selection strategy addresses several critical requirements for developing a robust thermal human detection system. FLIR ADAS v2 @FREEFLIRThermal provides automotive-focused thermal imagery with high thermal contrast between human subjects and vehicle/road backgrounds, captured at various distances typical of roadside surveillance applications. The dataset's vehicle-mounted perspective and diverse geographic coverage (Santa Barbara, San Francisco, London, Paris, Spanish cities) ensures exposure to varying ambient temperatures and thermal background conditions that challenge model generalization.

The datasets used are specifically selected to form a comprehensive compound training dataset that aims to cover a large v *TODO: CONTINUE HERE - AND REPLACE TABLE DATA WITH CUSTOM SPLITS*

AAU-PD-T @hudaEffectDiverseDataset2020 contributes controlled pedestrian detection imagery with systematic annotation quality and consistent thermal signatures, establishing reliable benchmarks for model performance assessment. The elevated camera perspective and sports field environment provide scenarios with minimal thermal background clutter, enabling evaluation of pure human detection capabilities without complex environmental interference.

OSU-T @davisTwoStageTemplateApproach2005 delivers outdoor thermal surveillance scenarios with natural environmental temperature variations and diverse background thermal signatures that challenge model generalization capabilities. The university campus setting provides realistic pedestrian detection scenarios with varying crowd densities and complex thermal backgrounds from buildings, vegetation, and infrastructure.

M3FD @liuTargetawareDualAdversarial2022 offers thermal imagery with varying ambient conditions where human thermal signatures exhibit different polarities relative to background temperatures. The dual-modal nature of this dataset, while primarily used for its thermal component, ensures exposure to challenging scenarios where traditional RGB-based assumptions may not apply to thermal domain characteristics.

KAIST-CVPR15 @hwangMultispectralPedestrianDetection2015 contributes the largest volume of thermal pedestrian data captured across different times of day, providing extensive variety in ambient thermal conditions that affect human-background contrast relationships. The urban traffic environment introduces complex thermal scenes with multiple heat sources (vehicles, pavement, buildings) that require sophisticated discrimination capabilities.

This multi-dataset approach creates a comprehensive compound training dataset that addresses the fundamental challenge of thermal domain adaptation for human detection models originally designed for RGB imagery. The combination ensures robust evaluation across varying thermal polarities (human-hot vs human-cool scenarios), environmental temperature conditions (day/night thermal crossover points), subject distances (near-field sports surveillance vs far-field automotive detection), and thermal contrast scenarios (high-contrast winter conditions vs challenging summer thermal equilibrium).

The unified annotation scheme maps all dataset-specific labels to a consistent class hierarchy: person, car, bicycle, animal, other vehicle, and background. This standardization enables seamless integration while preserving the diversity of thermal signatures and environmental conditions represented across the constituent datasets. The resulting compound dataset encompasses over 130,000 images with more than 250,000 annotated objects, providing sufficient scale and diversity for robust model training while minimizing dataset-specific biases that could compromise generalizability across diverse infrared imaging conditions encountered in real-world thermal surveillance deployments.

For training purposes, FLIR ADAS v2, AAU-PD-T, KAIST-CVPR15, and M3FD Detection contribute to the training set, while FLIR ADAS v2 and AAU-PD-T provide validation data. OSU-T serves exclusively as an independent test set, ensuring unbiased evaluation on data completely unseen during training. This split strategy maintains rigorous experimental integrity while maximizing the utilization of available annotated thermal imagery for model development.

== Model Implementation <model-impl>

The base architecture, #acr("SSD")300, is implemented with two distinct backbone networks: VGG16 and ResNet152. Each backbone is evaluated in two initialization scenarios: pretrained on ImageNet and trained from scratch only on thermal imagery.

Since the normalization applied to the input images is skewed by the preprocessing techniques mentioned above, the backbone networks are equipped with an additional #acr("BN") layer that allow the network to learn an optimal normalization for specific characteristics of their respective preprocessing setup.

Any specifics about the specific implementation details that are not mentioned in this documentation can be found in the #link("https://github.com/LukasFlorian/Thermal-Image-Human-Detection.git", "source code repository"), mostly in `/src/model/models.py`.

=== #acrf("VGG") Backbone Implementation<vgg-backbone>

#let ssd-vgg = [#acr("SSD")-#acr("VGG")16]
#let ssd-resnet = [#acr("SSD")-#acr("ResNet")152]

For the #acr("SSD") network with #acr("VGG")-16 backbone, the default anchor box configuration from the original implementation @liuSSDSingleShot2016 is used.
To reduce computational complexity, the #acr("FC") layers are removed and replaced with an additional two convolutions. Since those convolutions encompass fewer parameters than the original #acr("FC") layers, the pre-trained models use a subset of the #acr("FC") parameters and organize them in a dilated convolutional kernel. The very last #acr("FC") layer is entirely discarded, as it only generated the final class prediction when #acr("VGG") is used as an ImageNet classifier.

The backbone base network is followed by a smaller auxiliary network of four sequential pairs, each consisting of a 1x1 convolutional kernel halving the number of channels and a 3x3 convolutional kernel doubling the channels again, but without padding, such that it reduces the spatial dimensions by 2 pixels in each direction.

Prediction heads are positioned before each max-pooling layer, after both of the final convolutional layers of the base network, and after all four convolution pairs of the auxiliary network. Each prediction head consists of two convolutional layers; one for bounding box regression and one for class prediction @liuSSDSingleShot2016. These six prediction heads amount to a total of 8732 anchors for detection attempts.


A detailed visualization of the #ssd-vgg architecture is shown in @initial-vgg-image.

#figure(
  image("assets/vgg.png", width: 100%, fit: "cover"),
  caption: [The initial #ssd-vgg architecture],
)<initial-vgg-image>
\"ConvX_Y\" names a series of convolutional layers preceded by a maxpooling layer. The individual layers are not visualized as they do not affect feature map dimensions.

#figure(
  caption: [#ssd-vgg anchor boxes],
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    inset: 10pt,
    align: horizon,
    table.header(
      [*Feature Map*],
      [*Dimensions*],
      [*Scale*],
      [*Aspect Ratios*],
      [*Priors*],
      [*Total Priors*],
    ),

    [Conv4_3], [38 x 38], [0.1], [1:1, 2:1, 1:2, extra prior], [4], [5776],
    [Conv7], [19 x 19], [0.2], [1:1, 2:1, 1:2, 3:1, 1:3, extra prior], [6], [2166],
    [Conv8_2], [10 x 10], [0.375], [1:1, 2:1, 1:2, 3:1, 1:3, extra prior], [6], [600],
    [Conv9_2], [5 x 5], [0.55], [1:1, 2:1, 1:2, 3:1, 1:3, extra prior], [6], [150],
    [Conv10_2], [3 x 3], [0.725], [1:1, 2:1, 1:2, extra prior], [4], [36],
    [Conv11_2], [1 x 1], [0.9], [1:1, 2:1, 1:2, extra prior], [4], [4],
    [*Grand total*], [-], [-], [-], [-], [*8732*],
  ),
)<priors-vgg>
In this table, "Dims" is short for Dimensions for formatting reasons, the scale and aspect ratios combined result in the default scaling ratios for each respective prior and the extra prior has a 

=== #acrf("ResNet") Backbone Implementation<resnet-backbone>

The #acr("ResNet")-152 architecture does not entirely replicate that from the original paper @heDeepResidualLearning2015. Instead, it uses that from the PyTorch implementation, which is known as #acr("ResNet") v1.5 and has been shown to outperform the original architecture @ResNetV15PyTorch. The final #acr("FC") layer and average pooling layer are discarded, as they only serve the prediction in image classification, and are replaced by a single convolutional layer.

In the initial setup, prediction heads are placed on top of each of the major building blocks doubling the number of channels. Furthermore, one last auxiliary layer is added in order to attain an additional high-scale feature map. For the sake of simplicity, the single auxiliary layer is implemented as part of the #acr("ResNet") base network

As described in @vgg-backbone, the #ssd-resnet is also later adapted to apply #acr("BN") to the inputs to the prediction heads and to perform logistic regression using the sigmoid function by dropping the background class.

#figure(
  image("assets/resnet.png", width: 100%, fit: "cover"),
  caption: [The initial SSD-ResNet architecture],
)<initial-resnet-image>

\"LayerX\" in the architecture diagram names a series of convolutional bottleneck blocks - for a more detailed description of the individual layers and blocks, refer to @ResNetV15PyTorch and @backbone-networks. It is important to keep in mind that despite the #ssd-resnet diagram appearing less complex than that of #ssd-vgg, it is actually significantly deeper, as noted in the aforementioned @backbone-networks and also made apparent by compqaring @priors-resnet and @priors-vgg.

#figure(
  caption: [#ssd-resnet anchor boxes],
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    inset: 10pt,
    align: horizon,
    table.header(
      [*Feature Map*],
      [*Dimensions*],
      [*Scale*],
      [*Aspect Ratios*],
      [*Priors*],
      [*Total Priors*],
    ),

    [Layer1], [75 x 75], [0.05], [1:1, 2:1, 1:2], [3], [16875],
    [Layer2], [38 x 38], [0.1], [1:1, 2:1, 1:2, extra prior], [4], [5776],
    [Layer3], [19 x 19], [0.2], [1:1, 2:1, 1:2, 3:1, 1:3, extra prior], [6], [2166],
    [Layer4], [10 x 10], [0.375], [1:1, 2:1, 1:2, 3:1, 1:3, extra prior], [6], [600],
    [Aux_Layer], [5 x 5], [0.55], [1:1, 2:1, 1:2, 3:1, 1:3, extra prior], [6], [150],
    [*Grand total*], [-], [-], [-], [-], [*25567*],
  ),
)<priors-resnet>



== Later Developments <later-devs>

=== Sigmoid Activation Function <sigmoid-setup>
Initially, the activation function used for the prediction heads is the softmax function. However, since the network only needs to predict only two classes, peerson and background, the softmax function is later replaced with the sigmoid function from @sigmoid-eqn that outputs a single value between 0 and 1, representing the probability of the input being a person. This change is done in response to training results and is supposed to improve the performance of the network as it does not require the model to learn the more complex relationship between its outputs in the softmax function.


$
  "sigmoid"(x) = 1/(1+e^(-x))
$<sigmoid-eqn>@dubeyActivationFunctionsDeep2022

Additionally, later developed variants of the model utilize #acr("BN") layers in the prediction heads before passing on the input they get to the convolutional layers. The reasons for this will be explained based on the training results in @training-perf.

*
TODO: VGG + RESNET adaptations of:
- prior boxes
- test
*
// VGG: Later model configurations halve the number of output channels for the class prediction convolutions, effectively reducing the number of classes to 1, replace the softmax activation function with a sigmoid function and employ one #acr("BN") layer at the beginning of each prediction head. The reasons for this are explained in detail in @training-perf.

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
Compares #ssd-vgg and SSD-ResNer performance and discusses trade-offs between accuracy and computational efficiency.

== Practical Deployment Considerations <deployment>
Discusses real-world application scenarios and system requirements for thermal surveillance.

= Conclusion and Future Work <conclusion>

This thesis has systematically evaluated the application of Single Shot MultiBox Detector architectures for human detection in thermal imagery, providing empirical evidence for optimal model configurations in surveillance applications. The comprehensive analysis of 16 model variants across multiple thermal datasets has yielded practical insights for deploying neural networks in infrared-based security systems. The findings contribute to both academic understanding and industrial implementation of thermal computer vision technologies.

*Key areas to develop:*
- Key findings summary: Best-performing model configurations identified
- Methodological contributions: Systematic evaluation framework for thermal detection
- Practical implications: Guidelines for industrial thermal surveillance deployment
- Technical achievements: Successful adaptation of #acr("RGB") models to thermal domain
- Research limitations: Dataset constraints and environmental factors
- Future research directions: Advanced architectures and multi-modal approaches
- Industry impact: Potential applications beyond security surveillance
- Recommendations: Implementation guidelines for practitioners

= Appendix <Appendix>
== Figures <figures>


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