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

The increasing demand for automated surveillance systems in security-critical applications has driven significant advances in computer vision technologies. While conventional RGB-based surveillance systems remain prevalent, they face inherent limitations in challenging environmental conditions such as low-light scenarios, adverse weather, and nighttime operations. Thermal infrared imaging presents a compelling alternative, offering consistent detection capabilities independent of ambient lighting conditions and providing unique advantages for human detection through body heat signatures.

*Key areas to develop:*
- Context: Growing need for reliable 24/7 surveillance systems
- Problem: Limitations of RGB cameras in challenging conditions
- Solution: Advantages of thermal imaging for human detection
- Research gap: Need for optimized neural networks for thermal imagery
- Project scope: Evaluation of SSD models for thermal human detection
- Thesis structure: Overview of methodology and contributions
- Industrial relevance: Partnership with Airbus Defence & Space


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
Most object detection methods can be broadly categorized into two main approaches: traditional methods and deep learning-based methods. Traditional methods mainly rely on handcrafted features and sliding window techniques, while deep learning-based methods leverage convolutional neural networks (CNNs) or vision transformer (ViT) architectures to automatically learn features from data.

=== Traditional Object Detection Methods <traditional-methods>
Examines feature-based and sliding window techniques, such as Haar-like features and HOG descriptors.
=== Deep Learning-Based Object Detection <deep-learning-detection>
Discusses the evolution of deep learning models, including R-CNN, Fast R-CNN, Faster R-CNN, and YOLO, highlighting their strengths and limitations.


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

A #gls("Vulnerability") is a weakness in a system that can be exploited.

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

= Conclusion <final-conclusion>
