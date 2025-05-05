import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = r"C:\Users\Hp\Desktop\Makine-Ogrenmesi-Lablar\4.LinearRegression\Linear Regression - Sheet1.csv"
df = pd.read_csv(data)

X = np.c_[np.ones(len(df)), df["X"]]
Y = df["Y"].values.reshape(-1, 1)
m = len(Y)  # Örnek sayısı

theta = np.linalg.inv(X.T @ X) @ X.T @ Y

Y_pred = X @ theta

plt.scatter(df["X"], df["Y"], label="Gerçek Değerler")
plt.plot(df["X"], Y_pred, color='red', label="Lineer Regresyon")
plt.xlabel("Reklam Bütçesi")
plt.ylabel("Satış")
plt.legend()
plt.grid()
plt.title("En Küçük Kareler ile Lineer Regresyon")
plt.show()

print(f"Bias (b0): {theta[0][0]}")
print(f"Eğim (b1): {theta[1][0]}")

cost = (1 / (2 * m)) * np.sum((Y_pred - Y) ** 2)
print(f"Modelin Cost (MSE / 2) Değeri: {cost:.4f}")
