# Image_Captioning_DL
## Introduction
### How does the model work?
 This image captioning model follows an encoder-decoder architecture, where a CNN (Convolutional Neural Network) extracts visual
 features from an image, and an RNN(Recurrent Neural Network) generates a caption based on those features.
**Feature Extraction (Encoder - CNN):**
 A ResNet model processes the image and extracts visual features.
 The output is a feature vector representing the image content.
**Caption Generation (Decoder - RNN):**
 The extracted image features are passed to an RNN (LSTM) to generate a sequence of words.  The decoder predicts the next word in the caption based on the previous words and the image features.  The process continues until an end token is generated or a maximum length is reached.
**Training Process:**
 The model is trained using Cross-Entropy Loss, comparing the predicted words with actual captions.  The Adam optimizer updates the weights to minimize the loss.  The model learns to associate visual features with meaningful captions through multiple training iterations.
============================================================================================
### Key Features of the Approach
 *** CNN-RNN Pipeline:**
  Uses a ResNet encoder for feature extraction and an LSTM/GRU decoder for caption generation.

 *** Cross-Entropy Loss:**
  The loss function ensures that the predicted captions closely match ground truth captions.

 *** Training with Teacher Forcing:**
  During training, the model learns from actual captions instead of its own predictions to improve learning speed.

 *** GPU Acceleration:**
  The model runs efficiently on GPUs from Kaggle for faster training and inference.

 *** Checkpoint Saving:**
  The model periodically saves checkpoints, allowing training to resume if interrupted.

 *** Data Augmentation (Optional):**
  Techniques like image flipping or color adjustments can be used to improve generalization.
============================================================================================
## Model Architecture
**Encoder:* CNN (ResNet) extracts image features.
**Decoder:* RNN (LSTM) generates captions from extracted features.
**Loss Function:* Cross-Entropy Loss.
**Optimizer:* Adam.

**Why I chose ResNet over simple CNN?**
 1. It solves the Vanishing Gradient Problem, skipping conections preventing gradients from vanishing.
 2. Enables Training of Very Deep Networks
**Why I chose LSTM over simple RNN?**
 1. Also solves Vanishing Gradient Problem.
 2. Handles long sequences more efficiently.
 3. Reduces Exploding Gradients.
**Why I chose Cross-Entropy as Loss Function?**
 1. It's designed for classification problems. In our case multi-class classification.
 2. Works well with SoftMax.
============================================================================================
## How to Run the Code
All the code that needs to be run are in the imaeg_captioning.ipynb file.
You can run the code row by row.
============================================================================================
## Encountered Challenges
1. Kaggle Notebook interruption due to inactivity during model training.
*Solution:* Simulate user appearance. Run code in PyCharm that will type something into cell, in background.

2. Browser Shutting Down while Training, because of Lack of Memory in Browser.
*Solution:* I had to cut half of the dataset, so that training model won't take a lot of time.

3. Choosing right amount of layers in ResNet, to reduce time.
*Solution:* Tried ResNet50, but GPU couldn't handle it, so I decreased it to ResNet34.

4. Choosing Batch Size.
*Solution:* Took the maximum the GPU was capable of, which was 21.
============================================================================================
## Summmary
**Model Performance**
The model struggles to generate captions that are contextually relevant to the input images.
Achieved a BLEU score of 0.2>x.

**Limitations & Challenges**
The model tends to repeat common phrases, reducing caption diversity.
Struggles with uncommon objects or complex compositions.

**Key Learnings**
Adding attention mechanisms could improve word selection by focusing on relevant image areas.
More diverse datasets (e.g., MS COCO) could enhance caption variability and generalization.
Fine-tuning hyperparameters like learning rate and dropout significantly affects performance.
