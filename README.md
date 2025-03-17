
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

