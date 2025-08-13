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

A compelling alternative to systems operating in the visible light domain are such capturing wavelengths in the infrared spectrum and thus offering consistent detection capabilities that are fundamentally independent of ambient lighting conditions. Unlike regular RGB cameras, thermal sensors detect light with longer wavelengths that correspond to the heat signatures of emitted directly by objects. This characteristic provides unique advantages for human detection, as the human body maintains a relatively constant temperature of approximately 37°C, creating distinct thermal signatures that remain visible regardless of environmental illumination @akshathaHumanDetectionAerial2022.

The integration of deep learning architectures with thermal imaging thus opens new possibilities for automated systems that can reliably detect humans in the aforementioned scenarios with adverse conditions for conventional RGB-based concepts. However, most state-of-the-art object detection models have been primarily developed for and trained on RGB imagery. Given that the spectral, tectural and contrast characteristics of infrared images differ substantially from visible-light imagery, both due to the properties of those wavelengths themselves and of the sensors, those existing models might need to be adapted to achieve optimal performance.

//
This research addresses the critical need for systematic evaluation of neural object detection models specifically tailored for thermal human detection applications. The study focuses on the #acr("SSD") architecture, a prominent one-stage detection framework known for its balance between accuracy and computational efficiency. By examining multiple model variants with different backbone networks (VGG16 and ResNet152), initialization strategies (pretrained versus scratch training), and thermal-specific preprocessing techniques (image inversion and edge enhancement), this work provides comprehensive insights into optimal configurations for infrared surveillance systems.

This project addresses the need for systematic evaluation of neural object detection models and preprocessing techniques tailored for thermal human detection applications. For the purpose of developing and edge-deployable network, the focus of this study lies on the #acr("SSD") architecture, a prominent one-stage detection framework with relatively low complexity compared to newer architectures like the #acr("ViT").

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

+ *Dataset Integration Framework*: Unified evaluation approach across five diverse thermal datasets (FLIR ADAS v2, AAU-PD-T, OSU-T, M3FD, KAIST-CVPR15), enabling robust performance assessment.

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
However, those methods are either highly dependent on engineeri ng the correct priors, such as the Haar-like features, or limited to binary classification scenarios, as is the case for #acr("HOG")-based #acrpl("SVM"). Thus, newer Object Detection methods employ more complex deep-learning architectures. The best-performing models nowadays are #acrpl("ViT") using Attention mechanisms @dosovitskiyImageWorth16x162021 to learn relationships between patterns in different parts of images. However, they will not be further examined in this thesis, due to computational constraints that make them unfeasible for the edge-deployable solution sought in this work @dosovitskiyImageWorth16x162021.

Relevant for this examination are their predecessors, #acrpl("CNN"). The main mechanism they use to extract information from images are convolutional layers. Those convolutional layers get passed the image in the form of a tensor and perform matrix multiplication on that input tensor and a kernel tensor in a sliding window fashion to compute subsequent feature maps that will then be passed on as input to the next layer @lecunHandwrittenDigitRecognition1989.

At their core, these convolutional layers do not work inherently different from fully connected layers that compute several weighted sums across all components of the input tensor. More specifically, fully connected layers can be described as convolutional layers whose kernels have the same dimensions as the input tensor.

Resorting to smaller kernels, however, is a prior that makes use of the heuristic that in most cases, the features that compose an object in an image lie closely together. Thus, it is not necessary to process the entire image to detect an object that occupies only part of it. Convolutional neural nets hence save computational resources by focusing on smaller regions. In many cases it is advantageous to use those savings to increase network depth in order to make it possible for the network to learn more complex high-level features in subsequent layers.




- SSD rather than Faster R-CNN due to faster inference: @akshathaHumanDetectionAerial2022
- SSD > YOLO because of edge-specific faster inference: @alqahtaniBenchmarkingDeepLearning2024

// TODO: Incorporate switch to multi-label setup later

== Single Shot MultiBox Detector (SSD) Architecture <ssd-arch>
Detailed explanation of SSD model architecture, including backbone networks (VGG, ResNet) and detection mechanisms.

=== Backbone Networks for Feature Extraction <backbone-networks>
Explores the role of backbone networks (VGG, ResNet) in feature extraction and their impact on SSD performance.


=== Feature Maps and Anchor Boxes <feature-maps>
Describes the multi-scale feature maps and anchor boxes used in SSD for object detection.

=== MultiBox Loss Function <multibox-loss>
Explains the MultiBox loss function that combines localization loss and confidence loss for training SSD models.

* Non-Maximum Suppression (NMS): * <nms>

Examines the NMS technique used to filter duplicate detections and improve detection accuracy.


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

Just a couple of examples to demonstrate proper use of the typst template and its functions.

== Acronyms <acr-examples>

Use the `acr` function to insert acronyms, which looks like this #acr("HTTP").

#acrlpl("API") are used to define the interaction between different software systems.

#acrs("REST") is an architectural style for networked applications.

== Glossary <gls-examples>

Use the `gls` function to insert glossary terms, which looks like this:

The #gls("Stochastic Gradient Descent") is an optimization algorithm used in Machine Learning.

== Lists <list-examples>

Create bullet lists or numbered lists.

- This
- is a
- bullet list

+ It also
+ works with
+ numbered lists!

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

== References <ref-examples>

Cite like this #cite(form: "prose", <akshathaHumanDetectionAerial2022>).
Or like this @farooqObjectDetectionThermal2021.


You can also reference by adding `<ref>` with the desired name after figures or headings.

For example this @table references the table on the previous page.