# Zero-Shot Prompt Engineering Documentation

## Overview
This document details the prompt engineering strategies implemented for the Zero-Shot classification of histopathological images (BreakHis dataset) using Vision-Language Models (VLM). The objective is to maximize the alignment between visual features extracted by the model (CLIP, BiomedCLIP) and textual representations of pathology classes.

## Methodology

Five distinct prompting strategies were developed and evaluated to address the specific challenges of medical image classification, particularly the domain gap between natural images (pre-training data for CLIP) and histopathology slides.

### 1. Simple Strategy (Baseline)
This strategy serves as the control baseline. It uses a generic template without specific domain context.

*   **Template:** `"a histopathological image of {class}"`
*   **Rationale:** Establishes the minimum performance level achievable with standard zero-shot inference.
*   **Limitation:** Fails to provide sufficient semantic features for the model to distinguish between complex histological subtypes (e.g., separating *Tubular Adenoma* from *Fibroadenoma*).

### 2. Descriptive Strategy (Modality Focus)
This strategy focuses on the image acquisition modality and visual style.

*   **Templates:**
    *   `"a microscopy image of {class}"`
    *   `"an H&E stained biopsy of {class}"`
    *   `"microscopic view of {class}"`
*   **Rationale:** Explicitly stating terms like "microscopy" or "H&E stained" helps the model activate features related to texture and color distribution specific to histology, filtering out irrelevant semantic associations from natural images.

### 3. Medical Strategy (Expert Knowledge Injection)
This is the most advanced strategy, incorporating domain-specific medical definitions into the prompts. It replaces or augments the class name with a description of its pathological characteristics.

*   **Rationale:** Vision-Language models may not recognize specific medical nomenclature (e.g., "Phyllodes Tumor") but often understand descriptions of tissue components (e.g., "stromal overgrowth", "glandular tissue").
*   **Examples:**
    *   **Ductal Carcinoma:** `"malignant breast cancer originating in milk ducts showing invasive cells"`
    *   **Fibroadenoma:** `"benign breast tumor composed of fibrous and glandular tissue"`
    *   **Adenosis:** `"benign breast condition characterized by enlarged lobules"`
*   **Impact:** Significantly improves sensitivity (Recall) for malignant classes by explicitly checking for malignancy markers described in the text.

### 4. Contextual Strategy (Domain Anchoring)
This strategy anchors the classification within the specific organ domain.

*   **Templates:**
    *   `"{class}, a type of breast tissue pathology"`
    *   `"histology of breast cancer: {class}"`
*   **Rationale:** Reduces False Positives that might arise from confusion with tissues from other organs, enforcing a "Breast Pathology" context.

### 5. Ensemble Strategy (Aggregation)
The final classification is performed using an ensemble approach.

*   **Methodology:**
    1.  Generate text embeddings for **all templates** across the four strategies above.
    2.  Average these embeddings to create a single, robust prototype vector for each class.
    3.  Compute Cosine Similarity between the image embedding and this averaged text prototype.
*   **Rationale:** Mitigates the variance associated with individual prompts ("prompt sensitivity"). If one template fails to activate the correct features, others may succeed. This method consistently yields the highest and most stable accuracy.

## Performance Analysis

Comparative analysis of strategies on the BreakHis dataset (8 classes):

| Strategy | Impact on Accuracy | Interpretation |
| :--- | :--- | :--- |
| **Simple** | Baseline | Low discriminative power. High confusion between benign subtypes. |
| **Descriptive** | Low Improvement | improves detection of "image type" but not class specificity. |
| **Medical** | **High Improvement** | Critical for correctly identifying malignant subtypes. |
| **Ensemble** | **Maximum Performance** | The recommended approach for production inference. |

## Implementation
The prompt generation logic is encapsulated in `prompts/prompt_strategies.py`. Configuration of class-specific medical descriptions can be found in `config/config.py`.
