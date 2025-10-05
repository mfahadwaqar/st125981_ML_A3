# A3: Car Price Classification with Logistic Regression

**Student:** Muhammad Fahad Waqar  
**Student ID:** st125981  

### Preprocessing Pipeline
1. **Data Cleaning:**
   - Removed CNG/LPG fuel types and "Test Drive Car" owner category
   - Extracted numeric values from string columns (mileage, engine, max_power)
   - Extracted brand name from car model name
   - Mapped owner categories to numeric values (1-4)
   - Dropped `torque` column due to inconsistent formatting

2. **Feature Selection:**
   - Numeric: `year`, `km_driven`, `mileage`, `engine`, `max_power`
   - Categorical: `brand`, `transmission`, `owner`

3. **Encoding & Scaling:**
   - Imputed missing values with median (numeric) and mode (categorical)
   - Standardized numeric features using `StandardScaler`
   - One-hot encoded categorical features (drop first category)

4. **Target Transformation:**
   - Applied `log(selling_price)` for stable training
   - Binned into 4 classes using quartiles:
     - Class 0: Lowest 25% of prices
     - Class 1: 25-50% percentile
     - Class 2: 50-75% percentile
     - Class 3: Top 25% of prices

### Final Feature Count
- Training: (7,229 samples, 40 features)
- Testing: (799 samples, 40 features)

---

## Model Implementation

### Multinomial Logistic Regression

**Architecture:**
```python
LogisticRegression(
    learning_rate=0.01,
    max_iterations=500,
    regularization='ridge',  # or None
    lambda_reg=0.1
)
```

**Key Methods:**
- `softmax()`: Converts logits to class probabilities
- `h_theta()`: Hypothesis function (predictions)
- `gradient()`: Computes cross-entropy loss and gradient with optional L2 penalty
- `fit()`: Gradient descent optimization
- `predict()`: Returns predicted class labels
- `predict_proba()`: Returns class probabilities

## Experiment Results

### MLflow Experiments Conducted
1. **Basic Logistic Regression** (no regularization)
2. **Ridge Logistic Regression** (λ=0.1)
3. **Learning Rate Tuning** (0.001, 0.01, 0.1)
4. **Regularization Strength** (λ=0.01, 0.1, 1.0)
5. **Max Iterations** (100, 500, 1000)

### Best Model Performance
- **Model:** Logistic Regression with learning_rate=0.1
- **Run ID:** `6fb48e66e798477083264e84b5977dc2`
- **Metrics:**
  - Accuracy: 65.21%
  - Weighted F1: 0.6502
  - Weighted Precision: 0.6504
  - Weighted Recall: 0.6521

### Comparison: Custom vs Sklearn
| Metric | Custom Implementation | Sklearn |
|--------|----------------------|---------|
| Accuracy | 60.3% | 70.0% |
| Macro F1 | 0.582 | 0.70 |
| Weighted F1 | 0.581 | 0.70 |


