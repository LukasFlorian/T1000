# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains the thesis work "Evaluation of Neural Object Detection Models for Detecting Humans in Infrared Images" - a T1000 practical project for DHBW Ravensburg Campus Friedrichshafen in partnership with Airbus Defence & Space GmbH. The work focuses on developing AI-supported camera surveillance systems for human detection in infrared images.

## Document Structure

### Main Thesis Document
- **Main file**: `main.typ` - Primary Typst document containing the thesis content
- **Compilation**: `typst compile main.typ` - Generates `main.pdf` from the Typst source
- **Watch mode**: `typst watch main.typ` - Auto-compiles on file changes

### Supporting Files
- **Bibliography**: `sources.bib` - BibTeX references for citations
- **Acronyms**: `acronyms.typ` - Defines acronyms used throughout the document (API, HTTP, REST)
- **Glossary**: `glossary.typ` - Technical terms and definitions (Vulnerability, Patch, Exploit)

## Development Workflow

### Document Compilation
The project uses Typst as the document preparation system:
- Primary compilation: `typst compile main.typ` produces the final PDF
- Development mode: `typst watch main.typ` for continuous compilation during editing
- Output: `main.pdf` (committed to repository for sharing)

### Project Phases Structure
The thesis follows a structured approach across two practical phases:

**Phase 1**: Model evaluation and adaptation
- AI model evaluation (YOLO, SSD)
- RGB to infrared image adaptation
- Image transformation measures

**Phase 2**: Implementation and deployment
- SSD300-ResNet152 vs SSD300-VGG16 comparison
- Edge system development for surveillance
- Secure transmission system design

## Template Configuration

### Document Metadata
The thesis uses the `supercharged-dhbw` Typst template with these key configurations:
- **Language**: English (`language: "en"`)
- **Type**: "PROJEKTARBEIT T1000"
- **Author**: Lukas Florian Richter (TIK24, Computer Science - Artificial Intelligence)
- **Company**: Airbus Defence & Space, Taufkirchen
- **Duration**: 16 weeks
- **Logos**: Airbus branding enabled, DHBW logo available but disabled

### Typography and Layout
- **Fonts**: Custom Montserrat and Open Sans fonts included in `fonts/` directory
- **Spacing**: 1.5em paragraph spacing configured
- **Headers**: Chapter display enabled with right logo
- **Assets**: Images stored in `assets/` directory (AIRBUS_Blue.png, DHBW_Logo.png, ts.svg)

## Content Management

### Citation and References
- Use `#cite(form: "prose", <reference>)` for prose citations
- Use `@reference` for inline citations
- References defined in `sources.bib` using BibTeX format
- Cross-references with `<ref>` labels and `@label` syntax

### Specialized Content
- **Acronyms**: Use `#acr("ACRONYM")` function with definitions from `acronyms.typ`
- **Glossary**: Use `#gls("term")` function with definitions from `glossary.typ`
- **Code blocks**: Use `sourcecode` environment for syntax-highlighted code examples
- **Figures/Tables**: Standard Typst figure environment with captions

## Document Guidelines

### Academic Standards
- Confidentiality statement disabled (`show-confidentiality-statement: false`)
- University partnership clearly identified
- Professional formatting aligned with DHBW requirements
- Company supervision acknowledged (René Loeneke)

### Version Control
- Main document (`main.pdf`) is version controlled for collaboration
- Source files (`main.typ`, `acronyms.typ`, `glossary.typ`, `sources.bib`) tracked
- DHBW documentation stored in `dhbw_docs/` for reference