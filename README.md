# Structural-Aware Handwriting Similarity Model

## Objective
Design a high-speed, geometry-first model for online Chinese handwriting recognition and scoring, optimized for:
- Structural similarity rather than exact matching
- Continuous accuracy percentages
- Educational and learning-game feedback
- Edge-device deployment

The system outputs both a character prediction and a similarity-based accuracy score reflecting how close a user’s drawing is to the canonical structure.

## Input Representation

### Raw Input
Each handwritten character is captured as a sequence of pen points:

$$P = \{(x_i, y_i, t_i, p_i)\}, \quad i = 1 \dots N$$

### Normalization
1. Douglas--Peucker stroke simplification
2. Resampling to fixed length $L$ (e.g. $L=128$)
3. Spatial normalization to $[0,1]^2$

### Per-Point Feature Vector
Each point is encoded as:

$$f_i = [x_i, y_i, \Delta x_i, \Delta y_i, \sin\theta_i, \cos\theta_i, \kappa_i]$$

Final input tensor:

$$X \in \mathbb{R}^{L \times D}$$

## Model Architecture

### Backbone
- 1D Convolutional layers for local stroke primitives
- Single-layer GRU for temporal modeling
- Global average pooling

Output embedding:

$$E \in \mathbb{R}^{d}, \quad d \approx 128$$

## Dual-Head Output

### Character Classification
Softmax over $C$ characters:

$$\hat{y}_{char} \in \mathbb{R}^{C}$$

Loss:

$$\mathcal{L}_{char} = \text{CrossEntropy}(\hat{y}_{char}, y_{char})$$

### Structural Similarity Head
Predicted structural embedding:

$$\hat{y}_{struct} \in [0,1]^R$$

Cosine similarity:

$$S_{struct} = \frac{\hat{y}_{struct} \cdot y_{struct}}{\|\hat{y}_{struct}\|\|y_{struct}\|}$$

Structural loss:

$$\mathcal{L}_{struct} = 1 - S_{struct}$$

## Training Objective

$$\mathcal{L}_{total} = \mathcal{L}_{char} + \lambda \mathcal{L}_{struct}, \quad \lambda \in [0.05,0.2]$$

## Scoring for Learning Games
Final score:

$$Score = w_1 S_{struct} + w_2 S_{geom} + w_3 S_{prop}$$

| Score | Feedback |
| :--- | :--- |
| $\geq 90\%$ | Excellent |
| $75$--$90\%$ | Good |
| $60$--$75\%$ | Acceptable |
| $<60\%$ | Needs practice |

## Model Compression
Iterative Magnitude Pruning removes 90--95% of weights, followed by weight rewinding and retraining, producing a sparse winning-ticket network suitable for edge deployment.

## Deployment
- Export to ONNX
- INT8 quantization
- Sub-millisecond inference on mobile CPUs
- Fully offline operation

## Summary
The proposed model treats handwriting as geometry rather than pixels, supports graded similarity scoring, enables radical-level feedback, and is efficient enough for real-time educational applications.

## Acknowledgements

### CASIA Online and Offline Chinese Handwriting Databases

The online and offline Chinese handwriting databases, built by the CASIA, are released for academic research free of cost under an agreement.

From 2020, application form is not required for downloading the datasets fro acedemic research. All data can be downloaded at the page Data Download.

Commercial use of the databases is subject to charge. For possible license of commercial use, please contact Cheng-Lin Liu. The database of commercial use is enlarged to contain the data of 1,220 writers and made detailed annotations for all the text pages.

**Conditions of Academic Use**

1. All samples in the databases under this agreement can only be used by the group of the named applicant and can only be used for research purpose. No samples can be used for any commercial purpose.
2. The Institute of Automation of CAS retains the copyright of all sample data in the databases.
3. Publications of research results on the database should be appropriately acknowledged. The recommended reference is below:

> C.-L. Liu, F. Yin, D.-H. Wang, Q.-F. Wang, CASIA online and offline Chinese handwriting databases, Proc. 11th International Conference on Document Analysis and Recognition (ICDAR), Beijing, China, 2011, pp.37-41.

**Contact:**
Cheng-Lin Liu (liucl@nlpr.ia.ac.cn), Fei Yin (fyin@nlpr.ia.ac.cn)
National Laboratory of Pattern Recognition (NLPR)
Institute of Automation of Chinese Academy of Sciences
