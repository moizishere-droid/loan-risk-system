# Model ONNX Conversion Report
# Project: Loan Risk Assessment System
# Author: Abdul Moiz
# Date: 9 March 2026
# Phase: 14 — Model Saving + ONNX Conversion

## 1. Objective
Convert the 3 final selected models from joblib format to ONNX (Open Neural Network Exchange) format for faster inference, better portability, and production deployment via ONNX Runtime in the FastAPI backend.

Why ONNX:
| Reason              | Detail                                             |
|---------------------|----------------------------------------------------|
| Speed               | ONNX Runtime faster than sklearn/lgbm at inference |
| Portability         | Run on any language — Python, C++, Java            |
| Production standard | Industry standard for model deployment             |
| API ready           | FastAPI loads ONNX models directly via onnxruntime |

## 2. Libraries Used
| Library              | Version | Purpose                       |
|----------------------|---------|-------------------------------|
| onnx                 | 1.15.0  | Core ONNX format              |
| onnxruntime          | 1.17.1  | Run ONNX models at inference  |
| onnxmltools          | 1.16.0  | Convert LightGBM → ONNX       |
| skl2onnx             | 1.15.0  | Convert sklearn models → ONNX |
| onnxconverter-common | 1.13.0  | Shared conversion utilities   |

## 3. Models Converted
| Task           | Original Model   | Original Format         | ONNX File             |
|----------------|------------------|-------------------------|-----------------------|
| Classification | cls_LightGBM     | cls_LightGBM.joblib     | cls_LightGBM.onnx     |
| Regression     | reg_RandomForest | reg_RandomForest.joblib | reg_RandomForest.onnx |
| Clustering     | cluster_GMM      | cluster_GMM.joblib      | cluster_GMM.onnx      |

## 4. ONNX Model Details
| Model            | File Size  | IR Version | Graph Inputs | Graph Outputs        | Nodes |
|------------------|------------|------------|--------------|----------------------|-------|
| cls_LightGBM     | 1.05 MB    | 7          | float_input  | label, probabilities | —     |
| reg_RandomForest | 20.85 MB   | 7          | float_input  | variable             | —     |
| cluster_GMM      | 1341 bytes | 7          | float_input  | label, probabilities | 13    |

Note on file sizes:
- cls_LightGBM = 1.05 MB — efficient boosting tree structure
- reg_RandomForest = 20.85 MB — 171 trees stored individually in ONNX — expected
- cluster_GMM = 1341 bytes — GMM is a simple parametric model

Note on GMM outputs:
- Returns both label (hard assignment) AND probabilities (soft membership)
- Enables API to return "70% High Value Borrower" instead of just a hard label

## 5. Inference Verification
Sample: First 5 rows of validation set

### cls_LightGBM
| Property              | Value                                    |
|-----------------------|------------------------------------------|
| Output names          | label, probabilities                     |
| Labels                | [0, 0, 0, 0, 0]                          |
| Default Probabilities | [0.0983, 0.0567, 0.1230, 0.0000, 0.0048] |

### reg_RandomForest
| Property     | Value                                       |
|--------------|---------------------------------------------|
| Output names | variable                                    |
| Predictions  | [17.4874, 7.3839, 7.4486, 16.6844, 11.2476] |

### cluster_GMM
| Property          | Value                                                                            |
|-------------------|----------------------------------------------------------------------------------|
| Output names      | label, probabilities                                                             |
| Labels            | [1, 1, 1, 1, 1]                                                                  |
| Sample soft probs | [[0.000, 1.000], [0.000, 1.000], [0.000, 1.000], [0.398, 0.602], [0.000, 1.000]] |

## 6. ONNX vs Original Match
| Model              | Original                                    | ONNX                                        | Match|
|--------------------|---------------------------------------------|---------------------------------------------|------|
| cls_LightGBM probs | [0.0983, 0.0567, 0.1230, 0.0000, 0.0048]    | [0.0983, 0.0567, 0.1230, 0.0000, 0.0048]    | True |
| cls_LightGBM preds | [0, 0, 0, 0, 0]                             | [0, 0, 0, 0, 0]                             | True |
| reg_RandomForest   | [17.4874, 7.3839, 7.4486, 16.6844, 11.2475] | [17.4874, 7.3839, 7.4486, 16.6844, 11.2476] | True |
| cluster_GMM labels | [1, 1, 1, 1, 1]                             | [1, 1, 1, 1, 1]                             | True |

Note on reg_RandomForest tiny difference:
- Original = 11.2475, ONNX = 11.2476
- Floating point rounding — completely normal
- Within tolerance atol=1e-3
- Not a problem for production

All 3 ONNX models verified and match original models

## 7. Saved Files
| File                              | Size       | Description               |
|-----------------------------------|------------|---------------------------|
| models_onnx/cls_LightGBM.onnx     | 1.05 MB    | Classification ONNX model |
| models_onnx/reg_RandomForest.onnx | 20.85 MB   | Regression ONNX model     |
| models_onnx/cluster_GMM.onnx      | 1341 bytes | Clustering ONNX model     |