import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay, accuracy_score, classification_report
import matplotlib.pyplot as plt

file_path = r"C:\Users\Hp\Desktop\2.2 lab1\naiveBayes\diabetes.csv"
df = pd.read_csv(file_path)

# bağımsız değişkenler X bağımlı değişken y
X = df.drop(columns=["Outcome"]).values 
y = df["Outcome"].values  # çıkış değeri

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)

# Gaussian naive bayes modeli
class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.priors = {}
        
        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = X_c.mean(axis=0)
            self.var[c] = X_c.var(axis=0)
            self.priors[c] = len(X_c) / len(X)
    
    def predict(self, X):
        return [self._predict(x) for x in X]
    
    def _predict(self, x):
        posteriors = {}
        for c in self.classes:
            prior = np.log(self.priors[c])
            likelihood = np.sum(-0.5 * np.log(2 * np.pi * self.var[c]) - ((x - self.mean[c]) ** 2) / (2 * self.var[c]))
            posteriors[c] = prior + likelihood
        return max(posteriors, key=posteriors.get)

# modeli eğitip test ettim
start_time = time.time()
model = GaussianNaiveBayes()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
end_time = time.time()
net_zaman = end_time - start_time

print(f"Geçen süre: {net_zaman}")

# accuracy ve sınıflandırma raporu
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Model Doğruluk Oranı: {accuracy:.2f}")
print("Siniflandırma raporu:")
print(report)

# karmaşıklık matrisini oluşturalım
cm = confusion_matrix(y_pred, y_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
