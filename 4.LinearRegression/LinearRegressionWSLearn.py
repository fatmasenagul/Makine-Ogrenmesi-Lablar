from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(r"C:\Users\Hp\Desktop\Makine-Ogrenmesi-Lablar\4.LinearRegression\Linear Regression - Sheet1.csv")
df.columns = df.columns.str.strip()  # Sütun isimlerini temizle

X = df[["X"]].values
Y = df[["Y"]].values
m = len(Y)

model = LinearRegression()
model.fit(X, Y)

Y_pred = model.predict(X)

plt.scatter(X, Y, label="Gerçek Değerler")
plt.plot(X, Y_pred, color="green", label="scikit-learn Regresyon")
plt.xlabel("girdi")
plt.ylabel("sonuc")
plt.legend()
plt.grid(True)
plt.title("Scikit-learn ile Lineer Regresyon")
plt.show()

print(f"Bias (b0): {model.intercept_[0]:.4f}")
print(f"Eğim (b1): {model.coef_[0][0]:.4f}")

cost = (1 / (2 * m)) * np.sum((Y_pred - Y) ** 2)
print(f"Modelin Cost (MSE / 2) Değeri: {cost:.4f}")
