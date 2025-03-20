# Image Captioning with Deep Learning

## Introduction

### How does the model work?
This image captioning model follows an encoder-decoder architecture, where a CNN (Convolutional Neural Network) extracts visual features from an image, and an RNN (Recurrent Neural Network) generates a caption based on those features.

#### **Feature Extraction (Encoder - CNN):**
- A ResNet model processes the image and extracts visual features.
- The output is a feature vector representing the image content.

#### **Caption Generation (Decoder - RNN):**
- The extracted image features are passed to an RNN (LSTM) to generate a sequence of words.
- The decoder predicts the next word in the caption based on the previous words and the image features.
- The process continues until an end token is generated or a maximum length is reached.

#### **Training Process:**
- The model is trained using Cross-Entropy Loss, comparing the predicted words with actual captions.
- The Adam optimizer updates the weights to minimize the loss.
- The model learns to associate visual features with meaningful captions through multiple training iterations.

---

## Key Features of the Approach

- **CNN-RNN Pipeline:** Uses a ResNet encoder for feature extraction and an LSTM/GRU decoder for caption generation.
- **Cross-Entropy Loss:** Ensures that the predicted captions closely match ground truth captions.
- **Training with Teacher Forcing:** The model learns from actual captions instead of its own predictions to improve learning speed.
- **GPU Acceleration:** The model runs efficiently on GPUs from Kaggle for faster training and inference.
- **Checkpoint Saving:** The model periodically saves checkpoints, allowing training to resume if interrupted.
- **Data Augmentation (Optional):** Techniques like image flipping or color adjustments can be used to improve generalization.

---

## Model Architecture

- **Encoder:** CNN (ResNet) extracts image features.
- **Decoder:** RNN (LSTM) generates captions from extracted features.
- **Loss Function:** Cross-Entropy Loss.
- **Optimizer:** Adam.

### **Why ResNet over a simple CNN?**
1. Solves the Vanishing Gradient Problem by using skip connections to prevent gradients from vanishing.
2. Enables the training of very deep networks.

### **Why LSTM over a simple RNN?**
1. Also solves the Vanishing Gradient Problem.
2. Handles long sequences more efficiently.
3. Reduces Exploding Gradients.

### **Why Cross-Entropy as the Loss Function?**
1. Designed for classification problems, which fits our multi-class classification task.
2. Works well with SoftMax.

---

## How to Run the Code

All the necessary code is in the `image_captioning.ipynb` file.
You can run the notebook cell by cell.

---

## Encountered Challenges

1. **Kaggle Notebook interruption due to inactivity during model training.**
   - **Solution:** Simulated user activity by running a PyCharm script that interacts with the notebook in the background.

2. **Browser shutting down due to memory limitations.**
   - **Solution:** Reduced dataset size by half to ensure training could proceed efficiently.

3. **Choosing the right number of layers in ResNet to balance performance and efficiency.**
   - **Solution:** Initially tried ResNet50, but the GPU couldn't handle it, so I switched to ResNet34.

4. **Selecting an appropriate batch size.**
   - **Solution:** Chose the maximum batch size the GPU could handle, which was 21.

---

## Summary

### **Model Performance**
- The model struggles to generate captions that are contextually relevant to the input images.
- Achieved a BLEU score of **< 0.2**.

### **Limitations & Challenges**
- The model tends to repeat common phrases, reducing caption diversity.
- Struggles with uncommon objects or complex compositions.

### **Key Learnings**
- Adding attention mechanisms could improve word selection by focusing on relevant image areas.
- More diverse datasets (e.g., MS COCO) could enhance caption variability and generalization.
- Fine-tuning hyperparameters like learning rate and dropout significantly affects performance.

