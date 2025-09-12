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

*TODO: Reconsider Objectives vs. Contributions, place later in the conclusion*

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

+ It then applies Haar-like features - simple rectangular patterns (e.g. edge, line, or center-surround detectors) - to rapidly identify potential regions of interest. Each feature’s value is derived by subtracting the sum of pixels in one rectangle from the sum in an adjacent rectangle, leveraging the integral image for efficiency.

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
  caption: [Residual layer composed of layers with functions $f$ and $g$ with a shortcut connection],
  gap: 10pt
)<residual-graphic>

If the function $F$ is a composite function $F(x) = f(g(x))$, then @residual-fn can be transformed as seen in @residual-derivative to attain the derivative of $accent(y, hat)$ with respect to $x$, which given by the chain rule from @chain-rule.
$
  accent(y, hat)'(x) &= F'(x) + x',\
  &= f'(g(x)) dot g'(x) + 1
$<residual-derivative>
Here, the derivative of $x$ with respect to itself is 1. In reality, $x$ would be the result of a previous layer's function $h(x_0)$, which implies for the derivative of $accent(y, hat)$ with respect to $x_0$ the relationship introduced in @derivative-chain.
$
  accent(y, hat)(x)' &= accent(y, hat)(h(x_0))',\
  &= accent(y, hat)'(h(x_0)) dot h'(x_0),\
  &= (F'(h(x_0)) + 1) dot h'(x_0),\
  &= f'(g(h(x_0))) dot g'(h(x_0)) dot h'(x_0) + h'(x_0)
$<derivative-chain>
It becomes apparent that introducing residual layers and identity shortcuts helps to mitigate the #acr("VGP") by preserving the gradient with respect to previous layers' outputs without multiplying them with many small derivatives of subsequent layers. $F(x)$ can be an arbitrarily deep function and the gradient with respect to $x_0$ will still be $h'(x_0)$ plus a term that is multiplied by the derivatives of the residual function $F(x)$. This is in contrast to the gradient in a standard #acr("CNN") where repeated application of the chain rule is likely to cause the gradient to vanish.

When the input and output dimensions of a residual layer do not match, a linear projection is used to match the dimensions of the input and output before adding the two together. This linear projection is typically implemented using a 1x1 convolutional layer @heDeepResidualLearning2015. Even though this practice introduces additional parameters that need to be learned and makes the identity shortcut actually not an identity function anymore, it is still beneficial to the training process, as the residual function can be used to significantly increase the network's depth while preserving the gradient flow.

=== Momentum to achieve more stable convergence behavior <momentum>

Momentum in gradient descent optimization acts as a memory mechanism that accumulates a moving average of past gradients, enabling the optimizer to build velocity in directions of consistent gradient flow while reducing oscillations in high-curvature regions of the loss landscape. The momentum update rule modifies the standard gradient descent by maintaining a velocity vector $v_t$ that combines the current gradient with previous momentum, as shown in @momentum-eqn. @nakerstGradientDescentMomentum2020 @sutskeverImportanceInitializationMomentum

$
v_t &= beta dot v_(t-1) + (1 - beta) dot nabla L(theta_(t-1))\
theta_t &= theta_(t-1) - eta dot v_t
$<momentum-eqn>

Where $beta$ represents the momentum coefficient (typically 0.9), $eta$ is the learning rate, $nabla L(theta_(t-1))$ is the gradient of the loss function with respect to parameters $theta$ at time step $t-1$, and $v_t$ is the velocity vector at time $t$. This formulation enables the optimizer to continue moving in directions where gradients consistently point, leading to faster convergence and improved stability during training. @nakerstGradientDescentMomentum2020 @sutskeverImportanceInitializationMomentum

=== Weight Decay to prevent overfitting <weight-decay>

Weight decay is a regularization technique that prevents overfitting by adding an L2 penalty term to the loss function that discourages large parameter values. This technique helps the model generalize better to unseen data by constraining the complexity of learned representations. The regularized loss function combines the original loss with an L2 norm penalty on the parameters, as shown in @l2-loss-eqn. @yunWeightDecayScheduling2020

$
L_"regularized"(theta) = L_"original"(theta) + frac(lambda, 2) ||theta||_2^2
$<l2-loss-eqn>

Where $L_"original"(theta)$ is the standard loss function, $lambda$ is the weight decay coefficient, and $||theta||_2^2 = sum_i theta_i^2$ represents the L2 norm of the parameter vector (tensor). Taking the gradient of this regularized loss function with respect to the parameters yields:

$
nabla L_"regularized"(theta) = nabla L_"original"(theta) + lambda dot theta
$<l2-gradient-eqn>

This derivative demonstrates that the L2 regularization adds a term proportional to the parameters themselves to the gradient. Substituting this regularized gradient into the standard gradient descent update rule produces the weight decay update formula shown in @weight-decay-eqn:

$
theta_t = theta_(t-1) - eta dot (nabla L_"original"(theta_(t-1)) + lambda dot theta_(t-1))
$<weight-decay-eqn>

Where $eta$ is the learning rate and $theta_t$ are the updated parameters at time $t$. The weight decay term $lambda dot theta_(t-1)$ effectively shrinks the parameters toward zero at each update step at a step size proportional to their current magnitude. This creates a "drag" effect, preventing any single parameter from becoming too large and dominating the model's predictions. This regularization is particularly important when training on limited datasets, as it reduces the model's tendency to memorize training examples through large weights reacting heavily to known data points and improves generalization to new data. @yunWeightDecayScheduling2020 @smithDisciplinedApproachNeural2018 


== Computational Overhead Reduction Techniques <computational-overhead>

Training deep neural networks for object detection tasks presents significant computational challenges, particularly when evaluating multiple model variants across diverse datasets. Modern deep learning frameworks and hardware architectures provide several optimization techniques that can substantially reduce training time and memory requirements while maintaining model performance. This section examines the key computational optimization strategies employed in this work, focusing on their theoretical foundations and practical applications to thermal image processing workflows.

=== #acrf("MPT") <mpt-theory>

#acrf("MPT") is a computational optimization technique that addresses the memory and computational constraints encountered during deep neural network training. This approach leverages the reduced precision arithmetic capabilities of modern hardware accelerators while maintaining the numerical stability required for effective model convergence @micikeviciusMixedPrecisionTraining2018.

The fundamental principle behind #acr("MPT") involves utilizing lower-precision #acr("FP16") representations for the majority of training operations while preserving higher-precision #acr("FP32") calculations for operations that require numerical stability. This selective precision approach enables significant reductions in memory usage (typically 50% or more) and computational time while maintaining training effectiveness equivalent to full-precision training. @micikeviciusMixedPrecisionTraining2018

During mixed-precision training, the forward pass and gradient computation operations are performed using #acr("FP16") precision to maximize memory efficiency and computational throughput. However, the model weights are maintained in #acr("FP32") precision, and gradient updates, although computed based on half-precision inference results, are applied to these full-precision parameters. 

To address potential gradient underflow issues that can arise from the limited dynamic range of #acr("FP16"), gradient scaling techniques are employed. The loss function is multiplied by a scaling factor before backpropagation, effectively shifting small gradient values into the representable range of #acr("FP16"). After gradient computation, the gradients are unscaled before applying updates to the #acr("FP32") model weights @micikeviciusMixedPrecisionTraining2018.

The implementation of #acr("MPT") is particularly beneficial for large-scale training scenarios where memory constraints and computational efficiency are critical factors.

=== #acrf("JIT") Compilation and Graph Optimization <jit-theory>

#acrf("JIT") compilation represents a runtime optimization strategy that transforms computational graphs and Python code into optimized machine code during execution, rather than relying solely on interpreted execution. This approach addresses the inherent performance overhead of dynamic programming languages like Python, which traditionally suffer from interpretation bottlenecks during intensive computational workloads. @anselPyTorch2Faster2024 @PyTorch2x

The fundamental principle of #acr("JIT") compilation involves analyzing the computational patterns during initial execution runs and generating optimized compiled code for subsequent iterations. This process eliminates Python overhead, optimizes memory access patterns, and enables hardware-specific optimizations that can significantly accelerate training throughput. Modern deep learning frameworks implement #acr("JIT") compilation through graph compilation strategies that analyze the computational graph structure and apply fusion operations, kernel optimization, and memory layout improvements. @anselPyTorch2Faster2024

In PyTorch, #acr("JIT") compilation is implemented through TorchScript and the `torch.compile` function, which can provide substantial speedups for models with repetitive computational patterns. The compilation process involves tracing or scripting the model's forward pass to create an optimized representation of the forwards graph, which can optionally be used to also create a backwards graph for gradient computation. This can be achieved by using the `aot_eager` backend for compilation, which uses AOTAutograd (Ahead-Of-Time Autodifferentiation) wherever it is possible to do so or inserts graph breaks for non-differentiable operations. @anselPyTorch2Faster2024

Specifically using PyTorch's `aot_eager` backend for compilation means additionally creating a backwards graph for gradient computation ahead-of-time as soon as the forwards graph has been generated using the AOTAutograd graph analysis and generation engine. @anselPyTorch2Faster2024

The benefits of #acr("JIT") compilation become particularly pronounced in iterative training scenarios where the same computational patterns are repeated thousands of times across training epochs. Object detection models like #acr("SSD") benefit significantly from this optimization due to their complex multi-scale feature extraction and anchor box processing operations, which involve repetitive convolutions and tensor manipulations that can be efficiently optimized through compilation. Geometric mean speedups of $2.27 times$ during inference and $1.41 times$ for training have been reported across 180+ models of in the paper introducing JIT compilation and AOTAutograd for PyTorch @anselPyTorch2Faster2024.

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

Thermal images fundamentally differ from visible-light imagery in several key aspects that directly impact neural network performance. Unlike #acr("RGB") images that capture reflected light, thermal cameras detect electromagnetic radiation in the infrared spectrum, creating images based on the heat signatures emitted by objects @farooqObjectDetectionThermal2021.

The most distinctive characteristic of thermal imagery is its independence from ambient lighting conditions. Since thermal cameras detect heat radiation rather than reflected light, they provide consistent imaging capabilities in complete darkness, fog, smoke, and other challenging environmental conditions where traditional #acr("RGB") systems fail @akshathaHumanDetectionAerial2022. This makes thermal imaging particularly valuable for surveillance applications.

However, thermal images also present unique challenges for object detection models originally designed for visible-light imagery. The spectral characteristics result in different texture patterns, contrast relationships, and edge definitions compared to #acr("RGB") images. Additionally, thermal sensors often produce images with lower spatial resolution and different noise characteristics, requiring specialized preprocessing approaches to optimize detection performance. @beyererCNNbasedThermalInfrared2018

=== Preprocessing Techniques for Thermal Detection <thermal-preprocessing>

To address the unique characteristics of thermal imagery and improve object detection accuracy, this study implements three primary preprocessing techniques: normalization, polarity inversion, and edge enhancement. These techniques are designed to adapt #acr("RGB")-trained models to thermal domain characteristics while preserving essential spatial and thermal information.

*Normalization* serves as the fundamental preprocessing step, ensuring consistent input scaling across all thermal images and enabling proper transfer of learned features from the #acr("RGB") domain for the pretrained model backbones. Following standard practice, input images are normalized using the ImageNet dataset statistics with channel-wise means of [0.485, 0.456, 0.406] and standard deviations of [0.229, 0.224, 0.225] @dengImageNetLargescaleHierarchical2009.

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

Subsequently, Sobel operators are applied to detect horizontal and vertical edges. The Sobel kernels $S_x$ and $S_y$ for horizontal and vertical edge detection are defined as @burnhamComparisonRobertsSobel1997:

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
$<edge-magnitude>@burnhamComparisonRobertsSobel1997

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

The experimental evaluation employs five complementary thermal datasets that collectively provide comprehensive coverage of diverse infrared imaging scenarios and human detection challenges. The strategic selection of these datasets addresses the fundamental objective of creating a robust, generalizable detection system capable of operating across varying environmental conditions, camera configurations, and thermal imaging scenarios that are representative of real-world surveillance applications. @dataset-metadata-table provides a general overview of the different datasets.

#figure(
  caption: [Comprehensive thermal dataset specifications and characteristics],
  table(
    columns: (auto, auto, auto, auto, 17%),
    inset: 6pt,
    align: (x, y) => if y == 0 {center + horizon} else if x == 0 {left + horizon} else {center + horizon},
    table.header(
      [*Dataset*],
      [*Resolution*],
      [*Environment*],
      [*Camera Setup*],
      [*Annotated Classes*]
    ),

    [FLIR ADAS v2 @FREEFLIRThermal], 
    [640×512], 
    [Automotive roads,\ urban/highway,\ day/night,\ adverse weather],
    [Vehicle-mounted,\ forward-facing],
    [Person\ Vehicles\ Other objects],

    [AAU-PD-T @hudaEffectDiverseDataset2020], 
    [640×480], 
    [Controlled outdoor sports fields,\ winter conditions],
    [Elevated stationary (9m height)],
    [Person],

    [OSU-T @davisTwoStageTemplateApproach2005], 
    [360×240], 
    [University campus,\ natural outdoor pedestrian areas],
    [Elevated stationary (building rooftop, 3 stories)],
    [Person],

    [M3FD Detection @liuTargetawareDualAdversarial2022], 
    [1024×768], 
    [Urban driving,\ challenging visibility conditions],
    [Vehicle-mounted dual-modal],
    [Person\ Vehicles\ Lamp],

    [KAIST-CVPR15 @hwangMultispectralPedestrianDetection2015], 
    [640×512], 
    [Urban pedestrian scenarios,\ day/night cycles],
    [Vehicle roof-mounted,\ ego-centric],
    [Person\ People\ Cyclist],
  ),
)<dataset-metadata-table>

Since this thesis is on the detection of humans, the annotations are reduced to contain only the person label. All other labels are either discarded or, in the case of KAIST-CVPR15, converted. Note that "Vehicles" and "Other Objects" in @dataset-metadata-table serve as placeholders for the actual classes that are not listed in detail for the aforementioned reasons.

*FLIR ADAS v2* @FREEFLIRThermal provides automotive-focused thermal imagery with high thermal contrast between human subjects and vehicle/road backgrounds, captured at various distances typical of roadside surveillance applications. The dataset's vehicle-mounted perspective and diverse geographic coverage (Santa Barbara, San Francisco, London, Paris, Spanish cities) ensures exposure to varying ambient temperatures and thermal background conditions that challenge model generalization. The default test split of the dataset is preserved for the sake of scientific comparability; however, since not all images contain objects after filtering out all classes except for the person label, empty images from the predefined train split are instead moved to the validation split.

*AAU-PD-T* @hudaEffectDiverseDataset2020 is a dataset of images captured from cameras mounted at a height of 9m and directed at a soccer field. Thus, the objects are rather small and a wide variety of environmental conditions is covered by the data, mimicking the characteristics of typical surveillance scenarios. The default test split is preserved. Since no predefined validation split is provided, all empty images from the default training split are discarded entirely.

*OSU-T* @davisTwoStageTemplateApproach2005 serves only as a benchmark for testing the models due to its relatively small size of just 284 images. This also ensures testing with data that is not only completely unseen by the models before but also entirely unrelated to the images they have been trained on.

The *M3FD* @liuTargetawareDualAdversarial2022 dataset is originally intended to be used as a benchmark as well. Since its value for benchmarking lies in the multi-modality, however, it is instead split into train, validation and test images for this project, with a 60%/20%/20% ratio. That way the varying ambient conditions of the thermal imagery in this dataset also contribute to the overall diversity of the training data. The images in the training split are selected such that they contain at least one object.

*KAIST-CVPR15* @hwangMultispectralPedestrianDetection2015 contributes the second-largest volume of thermal pedestrian data captured across different times of day, providing extensive variety in ambient thermal conditions that affect human-background contrast relationships. The urban traffic environment introduces complex thermal scenes with multiple heat sources (vehicles, pavement, buildings) that require sophisticated discrimination capabilities. It contains three classes: person, people and cyclist. Since the cyclist bounding boxes only encapsulate the humans and not the rest of the bicycles, all cyclist labels are converted to person labels. To avoid "confusing" the model with different annotation standard for crowds across datasets, all images containing people objects are dropped entirely. By default, the dataset only has a train and test split. Due to the changes applied to the labels, model performance on the test split cannot be used for comparison to the results that have been reported for other models in the literature. Thus, the test split can be further modified by selecting half of the test images for validation.


 @dataset-split-table provides an overview of the resulting splits across the differnt datasets.

#figure(
  caption: [Dataset split overview. Val stands for Validation],
  table(
    columns: (24%, 12.66%, 12.66%, 12.67%, 12.66%, 12.66%, 12.67%),
    inset: 6pt,
    align: (x, y) => if y == 0 {center + horizon} else if x == 0 {left + horizon} else {center + horizon},
    
    table.cell(rowspan: 2, [*Dataset*]),
    table.vline(stroke: 2pt),
    table.cell(colspan: 3, [*Images per split*]),
    table.vline(stroke: 2pt),
    table.cell(colspan: 3, [*Objects per split*]),

    [*Train*],
    [*Val*],
    [*Test*],
    [*Train*],
    [*Val*],
    [*Test*],

    [FLIR ADAS v2],
    // Images
    [8205],
    [3681],
    [3749],
    // Objects
    [50478],
    [4470],
    [12323],
    

    [AAU-PD-T], 
    // Images
    [706],
    [0],
    [1000],
    // Objects
    [5572],
    [0],
    [2237],
    

    [OSU-T], 
    // Images
    [0],
    [0],
    [284],
    // Objects
    [0],
    [0],
    [984],
    

    [M3FD Detection], 
    // Images
    [2520],
    [840],
    [840],
    // Objects
    [18231],
    [5803],
    [5739],
    

    [KAIST-CVPR15], 
    // Images
    [8815],
    [1909],
    [1787],
    // Objects
    [20024],
    [1661],
    [2186],
    table.hline(stroke: 2pt),
    [*Overall*],
    [20246], [6430], [7660],
    [94305], [11934], [23469],
    [*Relative*],
    [59,0%], [18,7%], [22,3%],
    [72,7%], [9,2%], [18,1%],
    [*Average number of objects per image*],
    [4,66], [1,86], [3,06]
  ),
)<dataset-split-table>

This multi-dataset approach creates a comprehensive compound training dataset that addresses the fundamental challenge of thermal domain adaptation for human detection models originally designed for RGB imagery. The combination ensures robust evaluation across varying thermal polarities (human-hot and human-cool scenarios), environmental temperature conditions (day/night thermal crossover points), subject distances (far-field sports surveillance and  near-field automotive detection), and thermal contrast scenarios (high-contrast winter conditions and challenging summer thermal equilibrium) @hudaEffectDiverseDataset2020, @hwangMultispectralPedestrianDetection2015 @liuTargetawareDualAdversarial2022. Additionally, the ratio between the different splits closely resembles the standard 60%/20%/20% distribution that is often utilized for computer vision applications. 

*TODO: split source*

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
,\"ConvX_Y,\" names a series of convolutional layers preceded by a maxpooling layer. The individual layers are not visualized as they do not affect feature map dimensions.

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

In the initial setup, prediction heads are placed on top of each of the major building blocks doubling the number of channels. Furthermore, one last auxiliary layer is added in order to attain an additional high-scale feature map. For the sake of simplicity, the single auxiliary layer is implemented as part of the #acr("ResNet") base network.

As described in @vgg-backbone, the #ssd-resnet is also later adapted to apply #acr("BN") to the inputs to the prediction heads and to perform logistic regression using the sigmoid function by dropping the background class.

#figure(
  image("assets/resnet.png", width: 100%, fit: "cover"),
  caption: [The initial #ssd-resnet architecture],
)<initial-resnet-image>

,\"LayerX,\" in the architecture diagram names a series of convolutional bottleneck blocks - for a more detailed description of the individual layers and blocks, refer to @ResNetV15PyTorch and @backbone-networks. It is important to keep in mind that despite the #ssd-resnet diagram appearing less complex than that of #ssd-vgg, it is actually significantly deeper, as noted in the aforementioned @backbone-networks and also made apparent by compqaring @priors-resnet and @priors-vgg.

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
Initially, the activation function used for the prediction heads is the softmax function. However, since the network needs to predict only two classes, person and background, the softmax function is later replaced with the sigmoid function from @sigmoid-eqn that outputs a single value between 0 and 1, representing the probability of the person class. This change is done in response to training results and is supposed to improve the performance of the network as it does not require the model to learn the more complex relationship between its outputs in the softmax function.


$
  "sigmoid"(x) = 1/(1+e^(-x))
$<sigmoid-eqn>@dubeyActivationFunctionsDeep2022

Additionally, later developed variants of the model utilize #acr("BN") layers in the prediction heads before passing on the input they get to the convolutional layers. The reasons for this will be explained based on the training results in @training-perf.

=== #ssd-resnet anchor box configuration
Additionally, the #ssd-vgg variations exhibit better training behavior than the #ssd-resnet models (for more details, refer to @training-perf) in their initial configuration. With a major architectural difference lying in the additional priors that #ssd-resnet applies on the feature maps of layer 1 with dimensions of 75x75, those priors are later removed, such that computational overhead is reduced and the learning process is guided towards higher-scale features of the subsequent layers.

Simultaneously, new priors with the aspect ratios 3:1 and 1:3 are added to the prediction head on top of layer 2.

=== #acr("BN") layers in prediction heads
Lastly, to improve convergence behavior and ensure proper feature scaling across all layers, the prediction heads of both the #ssd-resnet and #ssd-vgg models get #acr("BN") layers that normalize inputs before further processing to attain predictions.

== Training Procedure

The training methodology employs a systematic approach to optimize SSD models for thermal human detection across diverse environmental conditions. All models are trained using the combined dataset described in @dataset, utilizing #acr("SGD") optimization with carefully tuned hyperparameters to ensure convergence stability and optimal performance. The training process incorporates several advanced techniques to address computational constraints while maintaining model accuracy, including mixed-precision training for memory efficiency, model compilation for performance optimization, and adaptive learning rate scheduling for improved convergence behavior.

=== Core Training Configuration

The training procedure uses a consistent set of hyperparameters across all 16 model variants to ensure fair comparison between architectures and preprocessing techniques. The core training configuration is designed to balance convergence stability with computational efficiency:

#figure(
  caption: "Core Training Configuration",
  sourcecode[```py
    batch_size = 64
    learning_rate = 1e-4  
    epochs = 14
    momentum = 0.9
    weight_decay = 5e-4
    optimizer = SGD  # with differential bias learning rates
    ```],
)

The batch size of 64 provides adequate gradient estimation while remaining within memory constraints of the available hardware. The relatively conservative learning rate of $10^(-4)$ ensures stable convergence across different model architectures and initialization strategies, preventing divergence during the critical early training phases.

=== Learning Rate Scheduling and Optimization

The optimization strategy employs #acr("SGD") with momentum, incorporating differential learning rates for bias parameters to improve convergence behavior. Bias parameters are assigned twice the base learning rate ($2 times 10^(-4)$), an empirical adjustment found to improve training stability in preliminary experiments.

Weight decay regularization ($5 times 10^(-4)$) prevents overfitting by penalizing large parameter values, particularly important given the relatively limited thermal training data compared to standard #acr("RGB") datasets. The momentum coefficient of 0.9 provides acceleration through consistent gradient directions while damping oscillations in parameter updates.

=== #acrl("MPT")

Due to computational constraints, the training procedure employs several optimization techniques described in @computational-overhead.

=== #acrl("MPT")

The training implementation uses #acr("MPT") as described in @mpt-theory. The mixed-precision implementation uses PyTorch's autocast context manager with CPU backend configuration:

#figure(
  caption: "Mixed-Precision Training Configuration",
  sourcecode[```py
    torch.autocast(device_type="cpu", dtype=torch.float16)
    ```],
)

This approach achieves approximately 50% reduction in memory usage while maintaining numerical stability equivalent to full-precision training, enabling the comprehensive evaluation of all 16 model variants within available computational resources.

=== Model Compilation and Performance Optimization

To accelerate training throughput, the implementation leverages #acr("JIT") compilation as described in @jit-theory. The compilation uses PyTorch's ahead-of-time eager mode optimization:

#figure(
  caption: "Model Compilation Setup",
  sourcecode[```py
    torch.compile(model, backend="aot_eager")
    ```],
)

Model compilation provides an expected 1.5-2× speedup in training time by optimizing computational graphs and reducing Python overhead during forward and backward passes. The compilation strategy is particularly beneficial for the iterative nature of object detection training, where the same computational patterns are repeated across thousands of training iterations.

=== Memory-Efficient Training Pipeline

Given the computational demands of training 16 different model variants, the training pipeline employs several memory management strategies to enable comprehensive experimentation within hardware constraints. Models are trained sequentially rather than in parallel, with explicit memory cleanup between training runs:

#figure(
  caption: "Memory Management Strategy",
  sourcecode[```py
    torch.cuda.empty_cache()  # CUDA GPU memory
    torch.mps.empty_cache()   # Apple Silicon unified memory
    gc.collect()              # Python garbage collection
    ```],
)

This sequential approach prevents memory fragmentation and allows each model to utilize the full available memory during training, enabling larger effective batch sizes and more stable gradient estimation. The explicit cache management ensures that memory allocated by previous model training runs is properly released before beginning subsequent experiments.

=== Data Loading and Augmentation Pipeline

The training pipeline employs optimized data loading with 4 parallel worker processes to prevent I/O bottlenecks during training. Persistent workers are enabled to reduce the overhead of worker initialization across epochs, particularly beneficial given the multiple datasets being processed simultaneously.

Data augmentation is integrated directly into the `ObjectDetectionDataset` class, applying photometric distortions (brightness and contrast adjustments) and geometric transformations (random expansion, cropping, and horizontal flipping) to increase training data diversity. All images are resized to the standard SSD300 input resolution of 300×300 pixels with ImageNet normalization statistics:

#figure(
  caption: "ImageNet Normalization Statistics",
  sourcecode[```py
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    ```],
)

This normalization approach maintains compatibility with ImageNet-pretrained backbones while adapting to the unique characteristics of thermal imagery through the preprocessing techniques described in @thermal-preprocessing.

=== Gradient Management and Numerical Stability

Optional gradient clipping is implemented to handle potential gradient explosion during training, particularly relevant for the deeper ResNet152 architecture. When enabled, gradients are clipped element-wise to prevent destabilization:

#figure(
  caption: "Gradient Clipping Implementation",
  sourcecode[```py
    param.grad.data.clamp_(-grad_clip, grad_clip)
    ```],
)

The optimizer state is preserved across checkpoint loading operations, ensuring that momentum buffers and other optimization state variables remain consistent when resuming training from saved checkpoints. This preservation is crucial for maintaining convergence properties when training is interrupted or distributed across multiple sessions.

=== Training Statistics and Monitoring

Training progress is monitored through comprehensive loss tracking with both per-epoch and per-iteration granularity. Loss statistics are saved in CSV format within the `stats/loss/` directory, with separate files for each model configuration. This detailed tracking enables post-hoc analysis of convergence behavior and identification of potential training instabilities.

Validation evaluation is performed regularly during training to monitor generalization performance and detect overfitting. The evaluation frequency is balanced to provide meaningful feedback without significantly impacting training throughput, particularly important given the computational overhead of the MultiBox loss calculation across 8732 anchor boxes per image.

Model checkpoints are saved at regular intervals and preserve complete training state including model parameters, optimizer state, and training metadata. This comprehensive state preservation enables reproducible resumable training and facilitates ablation studies across different hyperparameter configurations.

== Experimental Design <exp-design>

The experimental methodology employs a comprehensive factorial design to systematically evaluate the impact of architectural choices and preprocessing techniques on thermal human detection performance. This approach enables rigorous comparison across multiple variables while controlling for confounding factors that could bias results. The experimental framework is designed to provide statistically meaningful insights into optimal configurations for real-world thermal surveillance applications.

=== Model Configuration Matrix

The experimental design evaluates 16 distinct model configurations through a systematic $2 times 2 times 4$ factorial arrangement examining three primary factors:

*Backbone Architecture*: Two architectures representing different design philosophies
- VGG16: Traditional sequential CNN with 138M parameters, emphasizing simplicity and proven effectiveness
- ResNet152: Deep residual network with skip connections, addressing vanishing gradient problems through 152 layers

*Initialization Strategy*: Two approaches to parameter initialization  
- Pretrained: Models initialized with ImageNet-pretrained weights, leveraging #acr("TL") from RGB domain
- Scratch: Random initialization following Xavier uniform distribution, training exclusively on thermal data

*Preprocessing Configuration*: Four thermal-specific enhancement strategies
- None: Baseline with standard normalization only
- Inversion: Thermal polarity correction addressing hot-white vs. cold-white scenarios  
- Edge Enhancement: Two-stage enhancement combining Gaussian blur and Sobel edge detection
- Combined: Integration of both inversion and edge enhancement techniques

This factorial design results in comprehensive model naming convention: `SSD-{VGG|ResNet}-{pretrained|scratch}-{preprocessing}`, enabling systematic analysis of interaction effects between architectural choices and preprocessing strategies.

=== Evaluation Methodology Framework

The evaluation framework employs multiple complementary metrics to provide comprehensive assessment of model performance across different operational scenarios. The multi-metric approach recognizes that optimal model selection depends on specific deployment requirements and operational constraints.

*Primary Detection Metrics*:
- *mAP\@0.5*: Standard PASCAL VOC metric using 0.5 IoU threshold for positive detection classification
- *MS COCO Style mAP*: 101-point interpolated average precision across IoU thresholds from 0.5 to 0.95 in 0.05 increments
- *Precision-Recall Curves*: Performance characterization across confidence threshold ranges from 0.01 to 1.0

*Computational Efficiency Metrics*:
- *Inference Speed*: Forward pass timing on target hardware configurations
- *Memory Usage*: Peak GPU/CPU memory consumption during inference
- *Model Size*: Parameter count and storage requirements for edge deployment

The evaluation parameters are configured to reflect real-world deployment scenarios:

#figure(
  caption: "Evaluation Parameters Configuration",
  sourcecode[```py
    min_score = 0.01           # Minimum confidence for detection consideration
    max_overlap = 0.45         # NMS IoU threshold for duplicate removal
    top_k = 200                # Maximum detections per image
    ```],
)

These thresholds balance detection sensitivity with false positive suppression, particularly important for surveillance applications where excessive false alarms reduce system utility.

=== Statistical Analysis Framework

To ensure robust conclusions, the experimental design incorporates several statistical rigor measures addressing the inherent variability in neural network training and evaluation.

*Reproducibility Controls*:
- Fixed random seeds for dataset splitting ensuring consistent train/validation/test divisions
- Deterministic training procedures where computationally feasible
- Comprehensive checkpoint preservation enabling exact training replication

*Performance Validation*:
- Cross-dataset evaluation using multiple thermal datasets to assess generalization
- Independent test set (OSU-T) completely unseen during training for unbiased evaluation
- Statistical significance testing for performance differences between model configurations

*Ablation Study Design*:
The factorial structure enables systematic ablation studies examining:
- Architecture effects: VGG16 vs. ResNet152 performance isolation
- Initialization impact: Transfer learning effectiveness across thermal domains  
- Preprocessing contribution: Individual and combined enhancement technique effects
- Interaction analysis: Synergistic effects between architectural and preprocessing choices

=== Dataset Stratification and Cross-Validation

The experimental design addresses dataset heterogeneity through strategic splitting that maintains representative sampling across diverse thermal imaging scenarios. The multi-dataset approach ensures robust evaluation across varying environmental conditions, camera configurations, and thermal contrast scenarios.

*Training Set Composition*:
- FLIR ADAS v2: 8,205 images (40.5% of training data) providing automotive-focused scenarios
- AAU-PD-T: 706 images (3.5%) contributing elevated surveillance perspectives  
- M3FD Detection: 2,520 images (12.4%) adding urban driving complexity
- KAIST-CVPR15: 8,815 images (43.5%) representing diverse pedestrian scenarios

This distribution ensures adequate representation of major thermal imaging scenarios while preventing any single dataset from dominating model behavior. The relatively balanced contribution from FLIR ADAS v2 and KAIST-CVPR15 provides stability in training while smaller datasets contribute specialized scenarios.

*Validation Strategy*:
- Primary validation: FLIR ADAS v2 and KAIST-CVPR15 subsets providing diverse feedback during training
- Cross-validation: Models evaluated across all available datasets to assess generalization
- Independent testing: OSU-T dataset serving as completely unseen evaluation set

=== Performance Benchmarking Protocol

The benchmarking protocol establishes standardized evaluation procedures ensuring fair comparison across all model configurations and enabling reproducible results for future research.

*Evaluation Pipeline*:
1. *Model Loading*: Checkpoint restoration with complete state preservation
2. *Data Preprocessing*: Application of model-specific preprocessing pipeline
3. *Inference Execution*: Batched prediction generation with timing measurement
4. *Post-processing*: NMS application with standardized parameters
5. *Metric Calculation*: Comprehensive evaluation using vectorized operations for efficiency

*Hardware Standardization*:
All experiments are conducted on consistent hardware configurations to eliminate performance variations due to computational differences. Device selection follows automatic GPU/MPS/CPU detection with consistent memory allocation and optimization settings.

*Timing Methodology*:
Inference speed measurements exclude data loading and preprocessing overhead, focusing on core model computation time. Multiple measurement rounds with warm-up iterations ensure stable timing estimates unaffected by initialization overhead.

=== Experimental Controls and Bias Mitigation

The experimental design incorporates several controls to minimize potential sources of bias and ensure valid conclusions about model performance differences.

*Training Controls*:
- Identical hyperparameters across all model configurations preventing confounding from optimization differences
- Sequential model training with memory cleanup preventing interference between experiments  
- Consistent data augmentation and normalization ensuring fair preprocessing comparison

*Evaluation Controls*:
- Standardized inference procedures eliminating implementation-dependent performance variations
- Consistent metric calculation using identical evaluation code across all models
- Reproducible random sampling for statistical analysis and significance testing

*Environmental Controls*:
- Controlled software environment with fixed library versions
- Consistent hardware utilization through sequential rather than parallel training
- Systematic checkpoint and logging procedures enabling post-hoc verification

This comprehensive experimental design provides the methodological foundation for drawing reliable conclusions about optimal thermal human detection configurations while maintaining the scientific rigor necessary for industrial application and academic contribution.


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