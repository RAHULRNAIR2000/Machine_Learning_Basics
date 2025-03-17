
# Machine Learning

Machine Learning is a branch of Artificial Intelligence (AI) that allows computers to learn from data without explicitly being programmed. Instead of writing fixed rules, we train models using examples (data).

## Types of Machine Learning
ML is broadly classified into three categories:

1. **Supervised Learning** – Learning from labeled data
2. **Unsupervised Learning** – Finding patterns in unlabeled data
3. **Reinforcement Learning** – Learning from rewards and punishments

## Supervised Learning (Detailed Explanation)
Supervised learning is the most common type of ML. In this approach:

- The model is provided with input (X) and the correct output (Y).
- The model learns the relationship between X and Y.
- After training, the model can predict Y for new X values.

### Supervised Learning Example: Predicting House Prices
Consider a dataset of houses with their square footage (X) and price (Y):

| Square Footage (X) | House Price (Y) |
|-------------------|----------------|
| 1000 | ₹50,00,000 |
| 1500 | ₹75,00,000 |
| 2000 | ₹1,00,00,000 |

The goal is to train a model so that when provided with a new square footage (e.g., 1800 sqft), it predicts the expected house price.

## Types of Supervised Learning
Supervised learning is divided into two main types:

| Type | Purpose |
|------|---------|
| Regression | Predict a continuous value (e.g., House Price) |
| Classification | Predict a category (e.g., Spam or Not Spam) |

### Regression (Predicting Continuous Values)
Regression is used when the output is numerical (e.g., price, temperature).

#### Example: Predicting House Prices
Formula: 
\[ Y = mX + b \]

#### Popular Regression Algorithms:
- Linear Regression
- Polynomial Regression
- Decision Trees
- Random Forest
- XGBoost

### Classification (Predicting Categories)
Classification is used when the output is a category or label (e.g., "Yes" or "No", "Spam" or "Not Spam").

#### Example: Email Spam Detection
- **Input (X):** Email text
- **Output (Y):** Spam (1) or Not Spam (0)

#### Popular Classification Algorithms:
- Logistic Regression (for binary classification)
- Decision Trees
- Random Forest
- Support Vector Machines (SVM)
- Neural Networks

## Formula to Adjust Parameters (Gradient Descent)
To optimize the model, we adjust the values of \( m \) and \( b \) using gradient descent:

\[ m = m - \alpha \times \frac{\partial}{\partial m} (Error) \]
\[ b = b - \alpha \times \frac{\partial}{\partial b} (Error) \]

Where \( \alpha \) (Learning Rate) controls how big the adjustments are.

## Making Predictions
Once the model has found the best \( m \) and \( b \), it can predict values.

**Suppose the model learns:**
\[ Y = 50000X + 200000 \]

For \( X = 1800 \) sqft:
\[ Y = 50000(1800) + 200000 \]
\[ Y = ₹90,00,000 \]

So, the predicted price for an 1800 sqft house is ₹90,00,000.

## Evaluating Model Performance
After training, we check how good the model is using evaluation metrics.

### Common Evaluation Metrics:
- **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual values.
  \[ MSE = \frac{1}{N} \sum (Y_{actual} - Y_{predicted})^2 \]
- **R² Score (Coefficient of Determination)**: Measures how well the model explains the variance in data. Closer to 1 indicates a better model.

## Handling More Complex Data
### What if the relationship is NOT linear?
- Use Polynomial Regression or Neural Networks.

### What if there are multiple features (X1, X2, X3...)?
- Use Multiple Linear Regression:
  \[ Y = m_1X_1 + m_2X_2 + b \]

## Summary of What Happens Internally
1. The model assumes a mathematical function (e.g., \( Y = mX + b \)).
2. It learns the best values for \( m \) and \( b \) by minimizing the error using Gradient Descent.
3. Once trained, the model can predict \( Y \) for new \( X \) values.
4. The model is evaluated using metrics like MSE and R² score.



# Polynomial Regression & Neural Networks: Improving Accuracy

## Problem with Linear Regression
In Multiple Linear Regression, we assume a linear relationship:

\[ Y = a_1 X_1 + a_2 X_2 + ... + a_n X_n + b \]

However, in real-world scenarios (such as house price prediction), relationships are often non-linear.

### Example:
- A house with 2500 sqft may not cost exactly twice as much as a 1250 sqft house.
- Factors like exponential growth, saturation, or diminishing returns influence prices.
- Solution? Use **Polynomial Regression** or **Neural Networks** for better accuracy.

---

## 1. Polynomial Regression
### What is Polynomial Regression?
Polynomial Regression extends Linear Regression by introducing higher-degree (non-linear) terms:

\[ Y = a_1 X + a_2 X^2 + a_3 X^3 + ... + a_n X^n + b \]

**Advantages:**
- Captures curved relationships instead of just straight lines.
- Useful when price growth is exponential or non-linear.

### Example: Polynomial Regression in Python
#### Convert Features to Polynomial Form
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Create polynomial features (degree = 2)
poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())

# Train model
poly_model.fit(X_train, Y_train)

# Predict & Evaluate
Y_pred_poly = poly_model.predict(X_test)
r2_poly = r2_score(Y_test, Y_pred_poly)

print("Polynomial Regression R² Score:", r2_poly)
```

#### How Does It Help?
- **Degree 1 (Linear):** Straight Line (low accuracy for non-linear data)
- **Degree 2+:** Captures curves & complex patterns

---

## 2. Neural Networks for House Price Prediction
### How Do Neural Networks Work?
Neural Networks mimic the human brain and learn complex relationships using layers of neurons.

**Linear Regression Formula:**
\[ Y = a_1 X_1 + a_2 X_2 + ... + a_n X_n + b \]

Neural networks apply this multiple times using hidden layers with activation functions:

\[ Y = f(W_2 \cdot f(W_1 \cdot X + b_1) + b_2) \]

Where:
- **W₁, W₂** = weights (learned from data)
- **b₁, b₂** = biases
- **f(x)** = activation function (e.g., ReLU)

**Advantages:**
- Can learn highly non-linear relationships.
- Works well when data is large & complex.

### Neural Network Implementation for House Price Prediction
```python
import tensorflow as tf
from tensorflow import keras

# Define Neural Network
model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(5,)),  # Input Layer (5 features)
    keras.layers.Dense(10, activation='relu'),  # Hidden Layer
    keras.layers.Dense(1)  # Output Layer (House Price)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, Y_train, epochs=100, verbose=1)

# Predict
Y_pred_nn = model.predict(X_test)

# Evaluate
r2_nn = r2_score(Y_test, Y_pred_nn)
print("Neural Network R² Score:", r2_nn)
```

---

## Why Do Neural Networks Perform Better?
1. **Automatically Extracts Features**
   - Finds hidden patterns, like the effect of crime rate on house prices.
2. **Handles Non-Linear Relationships**
   - Adapts when factors impact prices differently at various levels.
3. **Learns Interactions**
   - Captures dependencies between features, such as square footage and location.

---

## Which One Should You Use?
| Model                 | Handles Non-Linearity? | Works on Large Data? | Performance |
|----------------------|----------------------|----------------------|------------|
| **Linear Regression** | No                 |  Yes               | OK (Simple) |
| **Polynomial Regression** |  Yes (Curved)    |  No (Overfits on large data) |  Good |
| **Neural Networks**   |  Best             |  Best            |  Amazing! |

---

## Conclusion
- **Use Polynomial Regression** when non-linearity exists but data is small.
- **Use Neural Networks** when dealing with large, complex datasets for higher accuracy.



