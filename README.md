<img width="226" height="161" alt="image" src="https://github.com/user-attachments/assets/e4a6a44d-e2ee-436d-b99f-531da8ec7d84" /># Project Report: CNN for Hand Sign Recognition

**Prepared by:** Soufiane KOUSTA
**Comparison with:** Anouar Moudad's Model  
**Requested by:** Ms. Asmae OUHMIDA



## Objective

The primary objectives are to:
- Develop an alternative CNN architecture with distinct design choices
- Improve training efficiency while maintaining competitive accuracy
- Analyze performance trade-offs between the two approaches

---

## Dataset Used

**Source:** ash2703/handsignimages from Kaggle Hub

**Dataset Statistics:**
- Training samples: 21,964 images
- Validation samples: 5,491 images  
- Test samples: 7,172 images
- Number of classes: 24 (A-Y, excluding J and Z)

<img width="955" height="985" alt="image" src="https://github.com/user-attachments/assets/3026f083-e316-4c36-9df3-8168b459c85d" />

---

## Data Analysis

### Image Dimensions and Batch Size

We use an image size of **40×40 pixels** and a batch size of **64**. This represents a 25% increase in resolution compared to the baseline (32×32) while maintaining fast training through larger batches.

### Data Augmentation

We use the following techniques to artificially expand the training dataset:

- **RandomFlip** - Horizontal flipping
- **RandomRotation** - Random rotation (±54°)
- **RandomZoom** - Random zoom (±15%)

This moderate augmentation strategy helps prevent overfitting while ensuring the model can still learn efficiently.

### Data Loading

We use `tf.keras.utils.image_dataset_from_directory`, which efficiently creates a `tf.data.Dataset` from a directory of images.

### Data Optimization

We apply several optimization techniques:

- **`.cache()`** - Keeps images in memory after first epoch, preventing disk I/O bottlenecks
- **`.shuffle(1000)`** - Randomizes the order of images to prevent sequence-based learning
- **`.prefetch()`** - Overlaps data preprocessing and model execution for better performance

These optimizations resulted in **~12 second** per-epoch training time.

---

## CNN Model Architecture

The model is a sequential `tf.keras.Model` with the following structure:

### Convolutional Blocks:
- **Block 1:** 2× Conv2D(40 filters) + BatchNormalization + MaxPooling + Dropout(0.3)
- **Block 2:** 2× Conv2D(80 filters) + BatchNormalization + MaxPooling + Dropout(0.4)
- **Block 3:** 2× Conv2D(160 filters) + BatchNormalization + MaxPooling + Dropout(0.4)

### Classification Layers:
- **Flatten** - Converts 3D feature maps to 1D
- **Dense(320)** - Fully connected layer with ReLU activation
- **Dropout(0.5)** - Regularization
- **Dense(24)** - Output layer with Softmax activation (24 classes)

### Layer Configuration:
All convolutional layers use:
- **Activation:** ReLU
- **Padding:** Same (preserves spatial dimensions)
- **Filter size:** 3×3

 
 <img width="318" height="364" alt="image" src="https://github.com/user-attachments/assets/ca2d7e16-d386-493e-99ff-d24c3050a344" />
<img width="226" height="161" alt="image" src="https://github.com/user-attachments/assets/3b064af5-7a2a-4d28-9496-6b6bc41018a9" />

---

## Model Compilation

The model is compiled with the following configuration:

- **Optimizer:** Adam (learning_rate=0.001)
- **Loss Function:** sparse_categorical_crossentropy
- **Metrics:** Accuracy

---

## Callbacks

Two callbacks are used to optimize training:

### EarlyStopping
- **Monitors:** val_loss
- **Patience:** 5 epochs
- **Restores best weights** when training stops
- **Purpose:** Prevents overfitting and saves training time

### ReduceLROnPlateau
- **Monitors:** val_loss
- **Factor:** 0.5 (halves learning rate)
- **Patience:** 3 epochs
- **Min LR:** 1e-6
- **Purpose:** Fine-tunes learning when validation loss plateaus

---

## Model Training

The model was trained with the following configuration:

- Training samples: 21,964 images
- Validation samples: 5,491 images
- Batch size: 64
- Maximum epochs: 30
- Callbacks: EarlyStopping and ReduceLROnPlateau

<img width="1728" height="652" alt="image" src="https://github.com/user-attachments/assets/ee974380-e55b-469d-a1db-462ead9a2c90" />

---

## Training Results

### Training Progress

The training history shows:

- **Rapid initial learning:** Model quickly learned basic patterns (6% → 93% accuracy in 5 epochs)
- **Early stopping at epoch 10:** Best validation accuracy (97.49%) achieved at epoch 9
- **Efficient convergence:** Training stabilized much faster than baseline (10 vs 30 epochs)
- **Fast training:** Average of **~12 seconds per epoch** (5× faster than baseline)

### Key Training Milestones:

| Epoch | Train Acc | Val Acc | Train Loss | Val Loss |
|-------|-----------|---------|------------|----------|
| 1 | 8.22% | 6.34% | 3.376 | 3.556 |
| 5 | 69.96% | 93.15% | 0.851 | 0.222 |
| 9 | 86.13% | **97.49%** | 0.402 | **0.075** |
| 10 | 87.79% | 95.94% | 0.359 | 0.118 |

**Best model saved at epoch 9 with 97.49% validation accuracy.**

---

## Model Evaluation

### Test Results

**Test Performance:**
- 113/113 - 7s - 61ms/step
- accuracy: 0.9424 - loss: 0.1600

### Final Results:

 **Test Accuracy: 94.24%**  
 **Test Loss: 0.1600**

### Analysis:

The model achieves strong performance on the test dataset:

- **94.24% accuracy** indicates the model correctly classifies most hand signs
- Moderate test loss (0.1600) shows reasonable confidence in predictions
- Performance demonstrates good generalization to unseen data
- **5× faster training** than baseline (12s vs 60s per epoch)

<img width="862" height="212" alt="image" src="https://github.com/user-attachments/assets/5741a25f-bd37-45cd-8526-0eadfb0cb0c2" />

---

## Training Visualization

<img width="874" height="320" alt="image" src="https://github.com/user-attachments/assets/7f739430-7cbf-47bd-9f3a-af62b8f0d84c" />

### Key Observations:

**Accuracy Curves:**
- Training and validation accuracy both increase steadily
- Validation accuracy peaks at 97.49% (epoch 9)
- Model converged quickly with early stopping

**Loss Curves:**
- Training and validation loss decrease consistently
- Best validation loss: 0.075 at epoch 9
- Minimal overfitting observed

**Convergence:**
- Model shows stable, rapid learning
- Early stopping prevented unnecessary training
- Training completed in ~2 minutes total

---

## Comparative Analysis

### Architecture Comparison

| Component | Original Model | Alternative Model | Difference |
|-----------|---------------|-------------------|------------|
| **Input Size** | 32×32 | **40×40** | +25% pixels |
| **Batch Size** | 32 | **64** | +100% |
| **Filters** | 32→64→128 | **40→80→160** | +25% |
| **Dense Units** | 256 | **320** | +25% |
| **Training Time/Epoch** | ~60s | **~12s** | **5× faster** |

### Performance Comparison

| Metric | Original Model | Alternative Model | Difference |
|--------|---------------|-------------------|------------|
| **Test Accuracy** | 99.55% | 94.24% | -5.31% |
| **Test Loss** | 0.0184 | 0.1600 | +0.142 |
| **Val Accuracy** | ~99.9% | 97.49% | -2.41% |
| **Training Time** | ~30 min | **~2 min** | **15× faster** |
| **Epochs Trained** | 30 | 10 | Early stopped |
| **Image Resolution** | 1,024 pixels | **1,600 pixels** | +56% |

### Key Findings

**Original Model Strengths:**
- ✅ Superior accuracy (99.55%)
- ✅ Lower test loss (0.0184)
- ✅ Smaller model size (~1.2M parameters)

**Alternative Model Strengths:**
- ✅ Much faster training (5× per epoch, 15× total)
- ✅ Higher resolution input (40×40 vs 32×32)
- ✅ Faster convergence (10 vs 30 epochs)
- ✅ Efficient for rapid prototyping

**Trade-offs:**
- The alternative model sacrifices 5.31% accuracy for 15× faster training
- Higher resolution captures more detail but requires more computation
- Ideal for research and experimentation; baseline better for production

---

## Use Case Recommendations

### When to Use Original Model (Anouar's):
- ✅ Production systems requiring maximum accuracy
- ✅ Critical applications (medical, accessibility)
- ✅ Final deployment where 99%+ accuracy is essential
- ✅ Resource-constrained devices (smaller model)

### When to Use Alternative Model:
- ✅ Rapid prototyping and experimentation
- ✅ Research projects with limited training time
- ✅ Applications where 94% accuracy is sufficient
- ✅ Situations requiring many training iterations

---

## Conclusion

The alternative CNN model successfully achieves **94.24% accuracy** on hand sign recognition with significantly improved training efficiency. The combination of:

- Higher resolution input (40×40 pixels)
- Balanced architecture (40→80→160 filters)
- Moderate data augmentation
- Efficient callbacks and optimization

Results in a model that trains **15× faster** than the baseline while maintaining good performance. This demonstrates the importance of understanding **accuracy-efficiency trade-offs** in deep learning.

**Final Verdict:**
- **For maximum accuracy:** Use original model (99.55%)
- **For rapid development:** Use alternative model (94.24%, 15× faster)

Both models have distinct advantages that make them suitable for different scenarios, showcasing practical considerations in model design and deployment.

---

**Summary Table:**

| Aspect | Winner | Key Advantage |
|--------|--------|---------------|
| Accuracy | Original | +5.31% better (99.55% vs 94.24%) |
| Training Speed | Alternative | 15× faster (2 min vs 30 min) |
| Image Quality | Alternative | 56% more pixels (40×40 vs 32×32) |
| Deployment Size | Original | Smaller, more efficient |
| Research Iteration | Alternative | Faster experimentation |
