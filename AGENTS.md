# AGENTS.md

## Project: Four-Leaf Clover Detection — Training to Mobile

## 1. Project Objective

Build an end-to-end computer vision system capable of detecting **four-leaf clovers in real time using a smartphone camera**.

The project consists of two major stages:

1. Train and evaluate a machine-learning model using the provided image/mask dataset.
2. Deploy the trained model into a mobile application capable of performing real-time inference from the device camera.

The final application should identify four-leaf clovers in natural clover fields and visually indicate their location on the camera preview.

---

## 2. Dataset

The project currently contains training and test data organized approximately as follows:

```text
dataset/
├── TrainImages/
├── TrainLabel/
├── TestImages/
└── TestLabel/
```

### Images

`TrainImages/` and `TestImages/` contain photographs of clover fields.

### Labels

`TrainLabel/` and `TestLabel/` contain binary segmentation masks.

The expected mask format is:

```text
BLACK  = background / non-target pixels
WHITE  = four-leaf clover pixels
```

The labels therefore represent **pixel-level segmentation**, not simple image classification.

Before training, inspect and validate the actual dataset structure rather than assuming filenames, extensions, dimensions, or naming conventions.

---

## 3. Primary ML Task

Treat the initial problem as **semantic segmentation**.

The model should receive an image:

```text
Input image
     ↓
Neural network
     ↓
Probability map
     ↓
Four-leaf clover segmentation mask
```

The system should ultimately be able to determine:

* whether a four-leaf clover exists;
* where it is located;
* its segmentation region;
* optionally its bounding box;
* confidence/probability.

Do not unnecessarily convert the dataset into another annotation format unless there is a demonstrated technical reason to do so.

---

## 4. Recommended Technology Stack

### Training

Use:

* Python
* PyTorch
* torchvision
* OpenCV
* NumPy
* Matplotlib
* Albumentations or an equivalent augmentation library
* CUDA when a compatible GPU is available

### Model

Start with a lightweight segmentation architecture suitable for eventual mobile deployment.

Candidates include:

* U-Net with a lightweight encoder
* MobileNet-based segmentation
* DeepLabV3 with a MobileNet backbone
* another lightweight architecture if experimentation demonstrates a clear advantage

Do not select a large model solely because it provides higher theoretical accuracy.

The final model must balance:

```text
Accuracy
+
Inference speed
+
Memory usage
+
Model size
+
Mobile compatibility
```

---

## 5. Development Philosophy

Work incrementally.

Do not immediately build the complete mobile application.

The preferred order is:

```text
Dataset inspection
        ↓
Dataset validation
        ↓
Baseline training
        ↓
Evaluation
        ↓
Model improvement
        ↓
Video inference on PC
        ↓
Model export
        ↓
Mobile inference
        ↓
Real-time camera integration
        ↓
Real-world testing
```

At every stage, maintain a working state.

Do not make large architectural changes without testing the previous implementation.

---

# 6. Phase 1 — Dataset Analysis

Before training any model, create tools/scripts to analyze the dataset.

Determine:

* number of training images;
* number of test images;
* image dimensions;
* mask dimensions;
* image formats;
* mask formats;
* number of channels;
* whether masks are truly binary;
* percentage of white pixels per mask;
* number of empty masks;
* minimum/maximum target size;
* whether every image has a corresponding label;
* whether filenames correctly match;
* whether train/test data are properly separated.

Generate useful visual samples showing:

```text
Original image
+
Corresponding mask
+
Mask overlaid on original image
```

Do not train until obvious dataset inconsistencies have been identified.

---

# 7. Phase 2 — Dataset Preparation

Implement a robust PyTorch Dataset/DataLoader pipeline.

Requirements:

* deterministic validation/testing;
* configurable image resolution;
* correct image/mask pairing;
* correct binary-mask handling;
* normalization;
* efficient loading;
* optional caching only if necessary.

Masks must use appropriate interpolation when resizing.

Never resize segmentation masks using standard image interpolation that introduces unintended intermediate labels.

---

# 8. Data Augmentation

Because the final application will operate in uncontrolled real-world conditions, augmentation should simulate realistic variation.

Consider:

* horizontal/vertical flips where appropriate;
* rotation;
* scale changes;
* crop;
* brightness changes;
* contrast changes;
* shadows;
* color variation;
* blur;
* noise;
* perspective changes.

Augmentations must be applied consistently to both image and mask when spatial transformations are involved.

Do not introduce unrealistic transformations merely to increase augmentation count.

---

# 9. Important Dataset Consideration

The model must not learn that a four-leaf clover is always present.

The dataset should contain negative examples whenever possible:

```text
Normal clover field
→ no four-leaf clover

Four-leaf clover field
→ four-leaf clover present
```

If the dataset does not contain sufficient negative examples, document this limitation and recommend a strategy for collecting additional data.

Also consider:

* partially occluded clovers;
* different lighting;
* different camera distances;
* different camera angles;
* dense clover fields;
* multiple four-leaf clovers;
* very small targets;
* visually similar normal clovers.

---

# 10. Phase 3 — Baseline Model

Implement a simple baseline first.

The baseline should establish:

* training loss;
* validation loss;
* IoU;
* Dice score;
* precision;
* recall;
* inference time.

Use a reproducible training configuration.

Save:

* model checkpoints;
* configuration;
* metrics;
* best model;
* training history.

The best model should be selected using validation performance rather than simply the final epoch.

---

# 11. Loss Function

Because the target object may occupy a small portion of the image, consider class imbalance.

Evaluate losses such as:

* Binary Cross Entropy;
* Dice Loss;
* BCE + Dice;
* Focal Loss if justified.

Do not assume one loss is universally optimal.

Document why the selected loss is being used.

---

# 12. Evaluation

Do not evaluate only on visual examples.

Report quantitative metrics:

```text
IoU
Dice
Precision
Recall
F1
```

Also evaluate performance specifically on:

* small four-leaf clovers;
* large four-leaf clovers;
* partially occluded clovers;
* difficult lighting;
* visually similar normal clovers;
* images containing no target.

Create visual evaluation examples:

```text
Original
Ground Truth
Prediction
Overlay
```

The test dataset must remain untouched during model development and hyperparameter tuning.

---

# 13. Error Analysis

When the model performs poorly, inspect the actual failure cases.

Classify errors such as:

```text
False positive
False negative
Poor localization
Incomplete segmentation
Confusion with normal clover
Lighting problem
Occlusion problem
Target too small
```

Use these observations to decide whether to:

* improve preprocessing;
* improve augmentation;
* modify the architecture;
* adjust the input resolution;
* collect additional data;
* modify the loss;
* improve labels.

Do not blindly increase model complexity.

---

# 14. Phase 4 — Real-Time PC Prototype

Before mobile deployment, create a real-time webcam/video inference prototype.

The prototype should:

1. capture camera frames;
2. preprocess the frame;
3. run model inference;
4. post-process the segmentation;
5. identify candidate regions;
6. draw the result on the frame;
7. display confidence;
8. report FPS and inference latency.

Example:

```text
Camera frame
      ↓
Model
      ↓
Segmentation mask
      ↓
Threshold
      ↓
Connected components / contours
      ↓
Candidate detection
      ↓
Overlay
```

This stage determines whether the model is actually useful for real-time detection.

---

# 15. Candidate Filtering

The raw segmentation output may contain noise.

Implement configurable post-processing where appropriate:

* probability threshold;
* minimum connected-component area;
* morphological operations;
* contour extraction;
* bounding-box generation.

Avoid hard-coded arbitrary thresholds.

Keep important thresholds configurable.

---

# 16. Phase 5 — Model Optimization

Once the model achieves acceptable accuracy, optimize it for mobile.

Investigate:

* reduced input resolution;
* lightweight architecture;
* ONNX export;
* ONNX Runtime;
* quantization;
* FP16;
* INT8 where supported;
* hardware acceleration.

Measure:

```text
Model size
RAM usage
Inference latency
FPS
Accuracy
```

Do not optimize prematurely.

Accuracy must remain acceptable after optimization.

---

# 17. Phase 6 — Mobile Application

Build a mobile application that:

* accesses the device camera;
* displays the live preview;
* performs inference locally on the device;
* does not require a cloud server for inference;
* overlays detected four-leaf clovers;
* displays useful confidence information;
* maintains acceptable real-time performance.

The preferred architecture is:

```text
Camera
   ↓
Frame
   ↓
Preprocessing
   ↓
Mobile ML Runtime
   ↓
Segmentation
   ↓
Post-processing
   ↓
Overlay
```

Inference should preferably happen **on-device** to minimize latency and avoid sending camera frames to a server.

---

# 18. Mobile Technology

Choose the mobile framework based on the actual deployment requirements.

Potential options:

### Android only

```text
Kotlin
+
Android Camera APIs
+
ONNX Runtime / compatible inference runtime
```

### Cross-platform

```text
Flutter
+
native camera integration
+
ONNX Runtime / appropriate mobile ML runtime
```

Do not introduce a cross-platform framework if it creates unnecessary ML integration complexity.

The priority is reliable real-time inference.

---

# 19. User Interface

Keep the first version simple.

The primary screen should contain:

```text
┌─────────────────────────────┐
│                             │
│        CAMERA VIEW          │
│                             │
│       ┌──────────┐          │
│       │  CLOVER  │          │
│       └──────────┘          │
│                             │
│  Confidence: 92%            │
│  FPS: 28                    │
│                             │
└─────────────────────────────┘
```

Avoid unnecessary features until detection works reliably.

---

# 20. Repository Structure

Prefer a structure similar to:

```text
four-leaf-clover-detector/
│
├── AGENTS.md
├── README.md
├── requirements.txt
│
├── data/
│   ├── TrainImages/
│   ├── TrainLabel/
│   ├── TestImages/
│   └── TestLabel/
│
├── src/
│   ├── dataset/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── inference/
│   └── utils/
│
├── scripts/
│   ├── analyze_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── export_model.py
│
├── checkpoints/
│
├── experiments/
│
├── notebooks/
│
└── mobile/
```

Adjust the structure if the actual project requirements justify it.

---

# 21. Reproducibility

Training must be reproducible as much as reasonably possible.

Record:

* random seed;
* model architecture;
* image resolution;
* batch size;
* learning rate;
* optimizer;
* scheduler;
* loss;
* augmentation configuration;
* number of epochs;
* dataset version;
* software dependencies.

Store experiment configurations rather than relying on undocumented command-line arguments.

---

# 22. Configuration

Avoid scattering constants throughout the source code.

Use a centralized configuration system for:

```text
DATASET_PATH
IMAGE_SIZE
BATCH_SIZE
LEARNING_RATE
EPOCHS
MODEL_NAME
NUM_WORKERS
DEVICE
THRESHOLD
CHECKPOINT_PATH
```

Make paths configurable so the project works on different machines.

---

# 23. Hardware

The training pipeline should automatically detect:

```text
CUDA GPU
      ↓
use GPU

otherwise
      ↓
use CPU
```

Do not assume a specific GPU.

The code must remain functional on CPU, even if training is significantly slower.

---

# 24. Code Quality

The agent must:

* write modular code;
* avoid unnecessary abstractions;
* use clear function/class names;
* add useful type hints where appropriate;
* handle errors explicitly;
* avoid hard-coded local paths;
* avoid duplicated logic;
* document non-obvious decisions.

Do not create excessive framework code for simple tasks.

Prefer simple, maintainable implementations.

---

# 25. Dependency Management

Every dependency must have a reason.

Before adding a package, consider whether the functionality can reasonably be implemented with existing dependencies.

Keep the environment reproducible.

Pin versions when necessary to prevent compatibility problems.

---

# 26. Testing

Create tests for critical functionality.

At minimum test:

* dataset loading;
* image/mask pairing;
* mask conversion;
* resizing;
* preprocessing;
* model output dimensions;
* post-processing;
* model export;
* inference.

A small synthetic dataset can be used for pipeline tests.

---

# 27. Git Practices

Use Git throughout development.

Commits should represent logical changes.

Examples:

```text
feat: add dataset validation
feat: implement segmentation baseline
feat: add training pipeline
feat: add evaluation metrics
feat: add webcam inference
feat: export model to ONNX
feat: add mobile inference
fix: correct mask resizing
```

Do not commit:

* large datasets;
* model checkpoints unless explicitly intended;
* credentials;
* API keys;
* local environment files;
* unnecessary generated files.

Provide an appropriate `.gitignore`.

---

# 28. Documentation

Maintain `README.md` throughout development.

The README should eventually explain:

1. Project objective.
2. Dataset structure.
3. Environment setup.
4. Training.
5. Evaluation.
6. Inference.
7. Model export.
8. Mobile application setup.
9. Performance results.
10. Known limitations.

Document commands so another developer can reproduce the project.

---

# 29. Agent Operating Rules

Before implementing a major feature:

1. Inspect the existing repository.
2. Inspect relevant files.
3. Understand the current architecture.
4. Reuse existing functionality when appropriate.
5. Make the smallest reasonable change.
6. Test the change.
7. Report what changed and how it was verified.

Never assume that a file, directory, dependency, model, or dataset property exists without checking.

When uncertain about the dataset, inspect it programmatically rather than guessing.

When an experiment fails, diagnose the failure before replacing the entire approach.

Do not prematurely optimize.

Do not prematurely build the mobile application.

---

# 30. Definition of Done

The project should ultimately satisfy all of the following:

### Dataset

* [ ] Dataset automatically validated.
* [ ] Image/mask correspondence verified.
* [ ] Dataset statistics generated.
* [ ] Training/validation/test methodology established.

### Model

* [ ] Segmentation baseline trained.
* [ ] Validation metrics reported.
* [ ] Test metrics reported.
* [ ] Failure cases analyzed.
* [ ] Best checkpoint saved.

### PC inference

* [ ] Single-image inference works.
* [ ] Video/webcam inference works.
* [ ] Four-leaf clovers are visually localized.
* [ ] FPS and latency are measurable.

### Deployment

* [ ] Model exported to a mobile-compatible format.
* [ ] Exported model produces results consistent with the original model.
* [ ] Model size and inference performance measured.

### Mobile

* [ ] Camera works.
* [ ] On-device inference works.
* [ ] Detection overlay works.
* [ ] Real-time performance measured.
* [ ] Application works without a network connection for inference.

### Documentation

* [ ] README complete.
* [ ] Installation documented.
* [ ] Training procedure documented.
* [ ] Evaluation results documented.
* [ ] Mobile deployment documented.
* [ ] Known limitations documented.

---

# 31. Most Important Principle

The goal is **not simply to achieve a high segmentation score on the provided dataset**.

The actual goal is:

> Build a model that can reliably find four-leaf clovers in real-world camera footage on a smartphone.

Therefore, prioritize **real-world generalization, false-positive control, small-object detection, inference speed, and mobile performance** alongside conventional ML metrics.
