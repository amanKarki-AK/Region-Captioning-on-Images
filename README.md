# Region-Based Image Captioning

[cite_start]This project is an implementation of the model described in the paper **"Deep Visual-Semantic Alignments for Generating Image Descriptions"** by Andrej Karpathy and Li Fei-Fei[cite: 4].

[cite_start]The primary goal of this project is to build a model that can generate natural language descriptions for specific regions within an image[cite: 13]. [cite_start]To achieve this, the model learns the complex correspondences between segments of text and the visual data they describe[cite: 14].

## Overall Approach & Flow

[cite_start]The model's approach is broken into a two-stage pipeline, as illustrated in Figure 2 of the paper[cite: 99]:

1.  [cite_start]**Alignment Model:** The first stage involves training a model to **infer alignments**[cite: 100]. [cite_start]This model learns to map segments of sentences (e.g., "a tabby cat") to the specific image regions (bounding boxes) they describe[cite: 74]. [cite_start]It is trained on a dataset of images and their corresponding full-sentence descriptions, treating the sentences as "weak labels"[cite: 40]. [cite_start]The output of this stage is a new dataset of inferred region-to-snippet correspondences[cite: 75, 100].

2.  [cite_start]**Generative Model:** The second stage uses the inferred correspondences from the first stage as training data[cite: 75]. [cite_start]A **Multimodal Recurrent Neural Network (RNN)** is trained to take an image region as input and generate a new, descriptive text snippet for that region[cite: 16, 75].

---

## Architectures Used

The project consists of two main neural network architectures, one for each stage.

### 1. The Alignment Model

[cite_start]This model learns to embed both image regions and sentence segments into a common multimodal space to measure their similarity[cite: 15, 45, 82].

* **Image Representation (RCNN):**
    * [cite_start]A **Region Convolutional Neural Network (RCNN)** is used to detect the top object locations (bounding boxes) in an image[cite: 104].
    * [cite_start]For each detected region, feature representations are extracted from a pre-trained CNN (from the fully connected layer just before the classifier)[cite: 109].

* **Sentence Representation (BRNN):**
    * [cite_start]A **Bidirectional Recurrent Neural Network (BRNN)** is used to process the full sentences[cite: 15, 119].
    * [cite_start]The BRNN computes a vector for each word that is enriched by its surrounding context from both the left and the right[cite: 122, 138].

* **Alignment Objective:**
    * [cite_start]The model is trained with a **structured, max-margin loss function**[cite: 170].
    * [cite_start]This objective function scores an image-sentence pair by summing the similarities of their best-aligned fragments[cite: 153]. [cite_start]It encourages correct image-sentence pairs to have a higher score than incorrect (mismatched) pairs by a margin[cite: 176].

### 2. The Generative Model

[cite_start]This model learns to generate text snippets conditioned on an input image region[cite: 198].

* **Multimodal Recurrent Neural Network (RNN):**
    * [cite_start]This architecture is an RNN-based language model that is conditioned on visual information[cite: 196, 198].

* **Generative Flow:**
    1.  [cite_start]An image (or image region) is passed through a CNN to extract its visual feature vector[cite: 201].
    2.  [cite_start]This image vector is fed into the RNN's hidden state **only at the first time step ($t=1$)** [cite: 201, 206][cite_start], typically as a bias[cite: 201].
    3.  [cite_start]A special `START` token is given as the first word input ($x_1$) to the RNN[cite: 210, 232].
    4.  [cite_start]The RNN then computes a probability distribution over the entire vocabulary to predict the first word ($y_1$)[cite: 205].
    5.  [cite_start]The predicted word is sampled (or the one with the highest probability is chosen) and becomes the input for the next time step ($x_2$)[cite: 211, 213].
    6.  [cite_start]This process is repeated, with the RNN generating one word at a time, until it predicts a special `END` token[cite: 211, 213, 232].
