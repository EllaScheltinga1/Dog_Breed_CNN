# Dog Breed Identification (Udacity Project)





---

## Project Overview

The notebook is structured in steps:

1. **Import datasets**
   - Dog images (train/valid/test) + labels for 133 breeds
   - Human images (LFW subset)

2. **Human detection**
   - Uses OpenCV Haar Cascades face detector to check for a visible face

3. **Dog detection**
   - Uses a pre-trained **ResNet-50** model trained on ImageNet  
   - Dog classes correspond to ImageNet indices **151–268**

4. **Breed classification (CNN from scratch)**
   - Custom CNN with multiple Conv + MaxPool blocks, Global Average Pooling, Dense + Dropout
   - Trained with checkpointing and learning-rate reduction

5. **Breed classification (Transfer Learning)**
   - Uses precomputed bottleneck features from a pre-trained CNN
   - Includes baseline VGG16 bottleneck model and a custom transfer learning head

6. **Final algorithm**
   - Combines human detector + dog detector + transfer learning breed classifier into one pipeline

---
