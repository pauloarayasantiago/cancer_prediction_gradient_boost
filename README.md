# Cancer Prediction Model 🏥

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0.2-orange.svg)](https://scikit-learn.org/)
[![GradientBoosting](https://img.shields.io/badge/GradientBoosting-Optimized-brightgreen.svg)](https://scikit-learn.org/stable/modules/ensemble.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview 📋

A machine learning solution for cancer diagnosis prediction achieving 94.6% recall rate, prioritizing early detection by minimizing false negatives. The model analyzes patient characteristics including genetic, lifestyle, and demographic factors to identify high-risk cases requiring immediate attention.

### Problem Statement
- **Challenge**: Early identification of cancer risk using patient data
- **Target**: Binary classification (0: No Cancer, 1: Cancer)
- **Dataset**: 1,500 patient records with 37.13% positive cases
- **Goal**: Maximize early detection while maintaining precision
- **Impact**: Enable timely intervention through accurate prediction

## Table of Contents
- [Data Insights & Analysis](#data-insights--analysis)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Future Work](#future-work)

## Data Insights & Analysis 📊

### The Story Behind the Numbers

Our analysis reveals a complex interplay of genetic, lifestyle, and demographic factors in cancer diagnosis. Here's what the data tells us:

#### 1. Primary Risk Factors

**Genetic Risk: The Strongest Predictor**
```
Risk Level    Population %    Diagnosis Rate    Risk Multiplier
Low          59.70%          28.9%             1.0x (baseline)
Medium       29.80%          41.2%             1.4x
High         10.50%          62.7%             2.2x
```
Key Finding: High genetic risk increases cancer likelihood by 2.2x compared to low risk, making it our strongest single predictor.

**Family History: The Genetic Connection**
```
Family History    Population %    Diagnosis Rate    Risk Impact
No History       85.00%          32.1%             1.0x (baseline)
Has History      15.00%          64.8%             2.0x
```
Key Finding: Family history doubles diagnosis probability, working synergistically with genetic risk.

#### 2. Lifestyle Impact

**The Protective Effect of Physical Activity**
```
Activity Level    Hours/Week    Diagnosis Rate    Risk Reduction
Low              0-3.3         45.2%             Baseline
Medium           3.3-6.6       35.8%             -20.8%
High             6.6-10.0      30.7%             -32.1%
```
Key Finding: High physical activity associates with 32.1% lower diagnosis rates, suggesting a strong protective effect.

**Smoking: A Major Modifiable Risk**
```
Status        Population %    Diagnosis Rate    Risk Increase
Non-Smoker    71.67%         31.2%            Baseline
Smoker        28.33%         52.4%            +68.0%
```
Key Finding: Smoking increases diagnosis rates by 68%, with stronger effects in high genetic risk groups.

#### 3. Demographic Patterns

**Age Distribution and Risk**
```
Age Group    Population %    Diagnosis Rate    Notable Patterns
20-40        35.2%          28.4%             Baseline risk
41-60        42.1%          39.7%             1.4x increase
61-80        22.7%          45.8%             1.6x increase
```
Key Finding: Risk increases with age but is moderated by lifestyle factors.

#### 4. Risk Combinations and Interactions

**High-Risk Combinations**
```
Risk Factor Combination               Diagnosis Rate
High Genetic Risk + Smoking          78.3%
Family History + Low Activity        71.2%
Age >60 + High Genetic Risk         69.5%
Multiple Risk Factors (3+)           82.4%
```
Key Finding: Risk factors compound multiplicatively rather than additively.

#### 5. Protective Factor Combinations

**Risk Reduction Patterns**
```
Protective Combination          Diagnosis Rate    Risk Reduction
High Activity + No Smoking     22.4%             -47.2%
Low Genetic Risk + High        18.9%             -55.3%
Activity
No Family History + High       20.1%             -52.5%
Activity
```
Key Finding: Multiple protective factors provide cumulative benefits.

### Feature Importance Analysis
```
Feature              Correlation    Notable Interactions
Genetic Risk         0.542         Strongest base predictor
Smoking              0.423         Amplifies genetic risk
Cancer History       0.389         Compounds with age
Physical Activity    -0.321        Moderates other risks
Age                  0.345         Increases impact of other factors
BMI                  0.284         Correlates with activity level
Alcohol Intake       0.152         Minor independent effect
Gender              0.089         Minimal impact
```

## Model Performance 📈

### Optimization Journey
```
Metric              Initial    After Optimization    Improvement
Accuracy            0.892      0.940                +5.4%
Precision           0.856      0.897                +4.8%
Recall              0.901      0.946                +5.0%
F1 Score            0.878      0.921                +4.9%
```

### Final Confusion Matrix
```
Predicted →    Negative    Positive
Negative       177         12        [93.7% True Negative Rate]
Positive       6           105       [94.6% True Positive Rate]
```

### Model Comparison
```
Model                  Accuracy    Recall
Gradient Boosting      0.925      0.946
Random Forest          0.919      0.932
SVM                    0.880      0.895
Decision Tree          0.869      0.878
Logistic Regression    0.849      0.856
k-NN                   0.834      0.845
Naive Bayes            0.826      0.834
```

## Installation 🛠️

```bash
# Clone repository
git clone https://github.com/[username]/cancer-prediction.git
cd cancer-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Unix
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements
```
python>=3.8
scikit-learn==1.0.2
scikit-optimize==0.9.0
pandas==1.5.3
numpy==1.23.5
matplotlib==3.7.1
seaborn==0.12.2
```

## Usage 💻

### Data Preparation
```python
from src.preprocessing import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor(
    continuous_features=['Age', 'BMI', 'PhysicalActivity', 'AlcoholIntake'],
    binary_features=['Gender', 'Smoking', 'CancerHistory']
)

# Process data
X_train, X_test, y_train, y_test = preprocessor.prepare_data(
    input_file='data/cancer_data.csv',
    test_size=0.2,
    random_state=42
)
```

### Model Training
```python
from src.models import CancerPredictionModel

# Initialize model with optimal parameters
model = CancerPredictionModel(
    n_estimators=176,
    learning_rate=0.085,
    max_depth=2,
    min_samples_split=5,
    min_samples_leaf=5
)

# Train model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

## Methodology 🔬

### Data Processing Pipeline
1. **Feature Engineering**:
   - Standardization of continuous features using StandardScaler
   - One-hot encoding for genetic risk levels
   - Binary encoding for categorical variables
   - Feature interaction terms for high-impact combinations

2. **Model Selection Process**:
   - Evaluated 7 classification algorithms
   - Selected Gradient Boosting based on recall priority
   - Optimized probability threshold (0.34) to minimize false negatives

3. **Hyperparameter Optimization**:
   ```python
   optimal_params = {
       'n_estimators': 176,
       'learning_rate': 0.085,
       'max_depth': 2,
       'min_samples_split': 5,
       'min_samples_leaf': 5
   }
   ```

### Model Architecture
The final model uses Gradient Boosting with:
- Binary cross-entropy loss function
- Custom probability threshold of 0.34
- Early stopping with 10-round patience
- Feature importance tracking
- Cross-validation with stratification

## Acknowledgments 🙏
- scikit-learn team for the machine learning framework
- Contributors and reviewers
