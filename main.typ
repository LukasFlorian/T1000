#import "@preview/supercharged-dhbw:3.4.1": *
#import "acronyms.typ": acronyms
#import "glossary.typ": glossary

#set par(spacing: 1.5em)
#show list: set block(spacing: 1.5em)

#let abstract = [test]

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

= Introduction
Introduces the problem of human detection in thermal images and the importance of infrared surveillance systems for security applications. Outlines the thesis objectives and structure.

= Literature Review and Theoretical Background
Reviews existing object detection methods, focusing on SSD architectures, and examines previous work on thermal image processing and human detection in infrared imagery.

== Object Detection Fundamentals
Covers basic principles of computer vision and object detection, including traditional methods and deep learning approaches.

== Single Shot MultiBox Detector (SSD) Architecture
Detailed explanation of SSD model architecture, inclusing backbone networks (VGG, ResNet) and detection mechanisms.

== Thermal Image Processing
Discusses characteristics of thermal images, preprocessing techniques (inversion, edge enhancement), and challenges specific to infrared imagery.

= Methodology
Describes the experimental setup, datasets used, model configuraitons, and evaluation metrics employed in the study.

== Dataset Description
Details the thermal image datasets (FLIR ADAS v2, AAU-PD-T, OSU-T, M3FD, KAIST-CVPR15) and their characteristics.

== Model Implementation
Explains the implementation of SSD models with different backbones and preprocessing configurations.

== Experimental Design
Outlines the systematic approach to comparing model variants and the evaluation framework.

= Results and Analysis
Presents comprehensive results from model training and evaluation, including performance comparisons across different configurations.

== Training Performance
Reports training loss curves, converegence behavior, and computational requirements for different mdoel variants.


== Detection Accuracy Analysis
Provides detailed mAP scores and detection performance metrics for each model configuraiton and preprocessing technique.

== Preprocessing Impact Evaluation
Analzyes the effects of image inversion and edge enhancement on detection performance.

= Discussion
Interprets the results, discusses the practical implications for surveillance systems, adn addresses limitations of the current approach.

== Model Performance Comparison
Compares SSD-VGG and SSD-ResNer performance and discusses trade-offs between accuracy and computational efficiency.

== Practical Deployment Considerations
Discuesses real-world application scenarios and system requirements for thermal surveillance.

= Conclusion and Future Work
Summarizes key findings, contributions to the field, and suggestst directions for future research in thermal image human detection.

= Examples

Just a couple of examples to demonstrate proper use of the typst template and its functions.

== Acronyms

Use the `acr` function to insert acronyms, which looks like this #acr("HTTP").

#acrlpl("API") are used to define the interaction between different software systems.

#acrs("REST") is an architectural style for networked applications.

== Glossary

Use the `gls` function to insert glossary terms, which looks like this:

A #gls("Vulnerability") is a weakness in a system that can be exploited.

== Lists

Create bullet lists or numbered lists.

- This
- is a
- bullet list

+ It also
+ works with
+ numbered lists!

== Figures and Tables

Create figures or tables like this:

=== Figures

#figure(caption: "Image Example", image(width: 4cm, "assets/ts.svg"))

=== Tables

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

== Code Snippets

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

== References

Cite like this #cite(form: "prose", <iso18004>).
Or like this @iso18004.

You can also reference by adding `<ref>` with the desired name after figures or headings.

For example this @table references the table on the previous page.

= Conclusion
