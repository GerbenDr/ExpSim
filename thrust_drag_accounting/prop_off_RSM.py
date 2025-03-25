import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import csv
from dataclasses import dataclass


class ResponseSurfaceModel:
    def __init__(self, degree=2):
        """
        Initialize the response surface model.
        :param degree: Degree of the polynomial (default is 2 for quadratic RSM).
        """
        self.degree = degree
        self.model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

    def fit(self, X, y):
        """
        Fit the response surface model to the data.
        :param X: Input features (2D array-like, e.g., [AoA, Vinf]).
        :param y: Output target (1D array-like, e.g., CL, CD, or CM25c).
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict the output using the fitted model.
        :param X: Input features (2D array-like, e.g., [AoA, Vinf]).
        :return: Predicted output values.
        """
        return self.model.predict(X)

    def get_coefficients(self):
        """
        Get the coefficients of the polynomial model.
        :return: Coefficients of the polynomial terms.
        """
        return self.model.named_steps['linearregression'].coef_

    def get_intercept(self):
        """
        Get the intercept of the polynomial model.
        :return: Intercept of the polynomial model.
        """
        return self.model.named_steps['linearregression'].intercept_

    def evaluate(self, X, y):
        """
        Evaluate the model using R-squared (coefficient of determination).
        :param X: Input features (2D array-like, e.g., [AoA, Vinf]).
        :param y: True output values.
        :return: R-squared score.
        """
        return self.model.score(X, y)

@dataclass
class PropOffData:
    AoA: float
    AoS: float
    Vinf: float
    CL: float
    CD: float
    CM25c: float

def read_prop_off_data(file_path):
    data_list = []
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter=';')
        for row in reader:
            # Convert values to appropriate types (replace commas with dots for floats)
            row_data = {
                'AoA': float(row['AoA'].replace(',', '.')),
                'AoS': float(row['AoS'].replace(',', '.')),
                'Vinf': float(row['Vinf'].replace(',', '.')),
                'CL': float(row['CL'].replace(',', '.')),
                'CD': float(row['CD'].replace(',', '.')),
                'CM25c': float(row['CM25c'].replace(',', '.'))
            }
            # Create an instance of PropOffData
            data = PropOffData(**row_data)
            data_list.append(data)
    return data_list

# Example usage
from pathlib import Path

file_path = Path(__file__).resolve().parent / "prop_off.csv"
prop_off_data = read_prop_off_data(file_path)
print(prop_off_data[0])


# Prepare the input features (X) and target (y)
X = np.array([[data.AoA, data.Vinf] for data in prop_off_data])  # Input features: AoA and Vinf
y_CL = np.array([data.CL for data in prop_off_data])  # Target: CL
y_CD = np.array([data.CD for data in prop_off_data])  # Target: CD
y_CM25c = np.array([data.CM25c for data in prop_off_data])  # Target: CM25c

# Create and fit the response surface model for CL
rsm_CL_po = ResponseSurfaceModel(degree=2)
rsm_CL_po.fit(X, y_CL)

# Create and fit the response surface model for CD
rsm_CD_po = ResponseSurfaceModel(degree=2)
rsm_CD_po.fit(X, y_CD)

# Create and fit the response surface model for CM25c
rsm_CM25c_po = ResponseSurfaceModel(degree=2)
rsm_CM25c_po.fit(X, y_CM25c)

# Predict using the fitted models
new_X = np.array([[5, 40], [15, 40]])  # Example input: [AoA, Vinf]
predicted_CL = rsm_CL_po.predict(new_X)
predicted_CD = rsm_CD_po.predict(new_X)
predicted_CM25c = rsm_CM25c_po.predict(new_X)
