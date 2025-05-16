LASSO REGRESSION – COMPLETE THEORY

------------------------------------------------------------

WHAT IS LASSO REGRESSION?

Lasso (Least Absolute Shrinkage and Selection Operator) is a linear regression technique that performs both regularization and feature selection.

It introduces a penalty term in the loss function to shrink coefficients — some of them may become exactly zero, eliminating irrelevant features.

------------------------------------------------------------

COST FUNCTION

Ordinary Linear Regression:

    J(θ) = (1/n) * Σ(y_i - ŷ_i)^2

Lasso Regression (with L1 penalty):

    J(θ) = (1/n) * Σ(y_i - ŷ_i)^2 + λ * Σ|θ_j|

Where:
- n: number of samples
- p: number of features
- ŷ_i: predicted value = X_i · θ
- λ: regularization strength (a hyperparameter)
- Σ|θ_j|: L1 penalty term (sum of absolute values of coefficients)

------------------------------------------------------------

INTUITION BEHIND LASSO

- Lasso adds a constraint that the sum of the absolute values of the model parameters is less than a fixed value.
- This causes some weights to become exactly zero, effectively removing features from the model.

Important: Lasso performs automatic feature selection!

------------------------------------------------------------

LASSO VS RIDGE REGRESSION

Feature              | Ridge (L2)                  | Lasso (L1)
---------------------|-----------------------------|-----------------------------
Penalty              | Sum of squares (θ_j^2)      | Sum of absolute values (|θ_j|)
Coefficients         | Shrink toward zero          | Some become exactly zero
Feature Selection    | No                          | Yes
When to Use          | All features are useful     | Only a few features are useful

------------------------------------------------------------

GEOMETRIC INTERPRETATION

- Lasso constraint region is a diamond (L1 norm).
- Ridge has a circular (L2 norm) constraint region.
- Corners of the diamond (Lasso) make it more likely for some coefficients to hit exactly zero.

------------------------------------------------------------

HOW REGULARIZATION AFFECTS THE MODEL

- λ = 0: Same as Linear Regression
- λ → ∞: All weights shrink to 0 (underfit)
- The best λ is found using techniques like cross-validation

------------------------------------------------------------

EFFECTS OF LASSO ON MODEL

- Reduces model complexity
- Prevents overfitting
- Improves generalization
- Selects relevant features

------------------------------------------------------------

EXAMPLE

Let’s say we have 5 features and the Lasso regression model finds:

    θ = [0.0, 2.3, 0.0, -1.1, 0.0]

This means:
- Feature 2 and 4 are useful
- Feature 1, 3, 5 were removed (coefficient = 0)

------------------------------------------------------------

FEATURE SELECTION

Lasso is often used for automatic feature selection when:
- You have many features
- You want a sparse model
- You want to know which features actually matter

------------------------------------------------------------

LASSO IN SCIKIT-LEARN (PYTHON)

from sklearn.linear_model import Lasso

model = Lasso(alpha=0.1)
model.fit(X_train, y_train)

print(model.coef_)
print(model.intercept_)

------------------------------------------------------------

LASSO PATH (ADVANCED)

We can visualize how the coefficients change as λ changes. This is called the Lasso path.

from sklearn.linear_model import lasso_path
alphas, coefs, _ = lasso_path(X, y)

# Plotting
import matplotlib.pyplot as plt
plt.plot(alphas, coefs.T)
plt.xlabel("Alpha")
plt.ylabel("Coefficients")
plt.title("Lasso Path")
plt.show()

------------------------------------------------------------

SUMMARY

- Lasso = Linear Regression + L1 Penalty
- Shrinks some coefficients to exactly zero
- Helps in feature selection
- Use it when your data has irrelevant or too many features
- Hyperparameter λ controls the amount of regularization

------------------------------------------------------------

NEXT: HANDS-ON IMPLEMENTATION

Move to the "02_basic" folder to practice Lasso using scikit-learn.

PRACTICE TIP:
Try Lasso on different datasets and observe how coefficients are eliminated.
