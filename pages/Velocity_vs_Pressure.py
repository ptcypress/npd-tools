# Re-import required libraries after code execution state reset
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.integrate import quad
from scipy.optimize import fsolve
import warnings

# Reload the CSV file
df = pd.read_csv("data/velocity_data.csv")
df.columns = df.columns.str.strip()

# Dynamically detect pressure and velocity columns
pressure_col = [col for col in df.columns if "Pressure" in col][0]
velocity_col = [col for col in df.columns if "Velocity" in col][0]

# Filter relevant brush types
df = df[df['Brush'].isin(['AngleOn™', 'Competitor'])]
angleon = df[df['Brush'] == 'AngleOn™']
competitor = df[df['Brush'] == 'Competitor']

# Prepare data
x_angleon = angleon[pressure_col].values.reshape(-1, 1)
y_angleon = angleon[velocity_col].values
x_comp = competitor[pressure_col].values.reshape(-1, 1)
y_comp = competitor[velocity_col].values

# Polynomial regression
poly = PolynomialFeatures(degree=3)
X_angleon_poly = poly.fit_transform(x_angleon)
X_comp_poly = poly.fit_transform(x_comp)

model1 = LinearRegression().fit(X_angleon_poly, y_angleon)
model2 = LinearRegression().fit(X_comp_poly, y_comp)

# Define functions
def f(x): return model1.predict(poly.transform(np.array([[x]])))[0]
def g(x): return model2.predict(poly.transform(np.array([[x]])))[0]
def diff(x): return f(x) - g(x)

# Solve for intersection
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        x_intersect = fsolve(diff, x0=1.0)[0]
    except:
        x_intersect = None

type(x_intersect), x_intersect
