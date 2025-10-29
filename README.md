# Region-Based Image Captioning

This project is an implementation of the model described in the paper **"Deep Visual-Semantic Alignments for Generating Image Descriptions"** by Andrej Karpathy and Li Fei-Fei.

The primary goal of this project is to build a model that can generate natural language descriptions for specific regions within an image.To achieve this, the model learns the complex correspondences between segments of text and the visual data they describe.

## Overall Approach & Flow

The model's approach is broken into a two-stage pipeline, as illustrated in Figure 2 of the paper:

1.  **Alignment Model:** The first stage involves training a model to **infer alignments**. This model learns to map segments of sentences (e.g., "a tabby cat") to the specific image regions (bounding boxes) they describe.It is trained on a dataset of images and their corresponding full-sentence descriptions, treating the sentences as "weak labels".
2.  The output of this stage is a new dataset of inferred region-to-snippet correspondences.

3.  **Generative Model:** The second stage uses the inferred correspondences from the first stage as training data. A **Multimodal Recurrent Neural Network (RNN)** is trained to take an image region as input and generate a new, descriptive text snippet for that region.

---

## Architectures Used

The project consists of two main neural network architectures, one for each stage.

### 1. The Alignment Model

This model learns to embed both image regions and sentence segments into a common multimodal space to measure their similarity.

* **Image Representation (RCNN):**
    * A **Region Convolutional Neural Network (RCNN)** is used to detect the top object locations (bounding boxes) in an image.
    * For each detected region, feature representations are extracted from a pre-trained CNN (from the fully connected layer just before the classifier).

* **Sentence Representation (BRNN):**
    * A **Bidirectional Recurrent Neural Network (BRNN)** is used to process the full sentences.
    * The BRNN computes a vector for each word that is enriched by its surrounding context from both the left and the right.

* **Alignment Objective:**
    * The model is trained with a **structured, max-margin loss function**.
    * This objective function scores an image-sentence pair by summing the similarities of their best-aligned fragments. It encourages correct image-sentence pairs to have a higher score than incorrect (mismatched) pairs by a margin.

### 2. The Generative Model

This model learns to generate text snippets conditioned on an input image region.

* **Multimodal Recurrent Neural Network (RNN):**
    * This architecture is an RNN-based language model that is conditioned on visual information.

* **Generative Flow:**
    1.  An image (or image region) is passed through a CNN to extract its visual feature vector.
    2.  This image vector is fed into the RNN's hidden state **only at the first time step ($t=1$)** , typically as a bias.
    3.  A special `START` token is given as the first word input ($x_1$) to the RNN.
    4.  The RNN then computes a probability distribution over the entire vocabulary to predict the first word ($y_1$).
    5.  The predicted word is sampled (or the one with the highest probability is chosen) and becomes the input for the next time step ($x_2$).
    6.  This process is repeated, with the RNN generating one word at a time, until it predicts a special `END` token.
