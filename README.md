# K-Nearest Neighbors & Radius Neighbors Hyperparameter Analysis

## 1. Purpose of Lab Work
This laboratory work explores the classification capabilities, accuracy patterns, and optimization boundaries of instance-based classifiers using the Sklearn Wine Dataset. By testing a series of distinct parameter sets across K-Nearest Neighbors (KNN) and Radius Neighbors (RNN), this experiment evaluates how local density metrics shape machine learning classifications.

## 2. Key Insights and Hyperparameter Trends
* **KNN Performance Matrix:** The classification engine achieves optimal precision at mid-range neighbor choices. Extremely low K values show minor overfitting vulnerabilities, while large K counts (K=21) introduce high bias by bleeding across class limits.
* **RNN Structural Realities:** The Radius Neighbors Classifier demonstrates high precision when the radius is well-tuned (between 1.5 and 2.5). If the radius is too small, points become completely isolated; if it expands too far, the bounding circle pulls in points from multiple classes, causing misclassifications.

## 3. Challenges Faced and Engineering Decisions
* **The Distance Variance Warning:** During initial execution, running an unscaled RNN model triggered a severe user warning highlighting that the outlier label was invalid due to empty neighborhoods. 
* **Resolution Strategy:** Because the Wine dataset features have completely different measurement ranges (e.g., Proline vs Alcohol), distance spheres were severely distorted. Introducing a `StandardScaler` to normalize features and adjusting the radius metrics down to small decimal spaces (1.0 to 3.5) successfully eliminated the execution crash.
