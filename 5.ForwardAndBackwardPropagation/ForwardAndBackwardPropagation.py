import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def relu(x):            return np.maximum(0, x)
def relu_deriv(x):      return (x > 0).astype(float)

def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def one_hot(y, n_classes):
    eye = np.eye(n_classes)
    return eye[y]

def accuracy(y_true, y_pred):        
    return (y_true == y_pred).mean()


class MLP:
    def __init__(self, n_in, n_hidden, n_out, lr=0.01, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2/(n_in+n_hidden)), size=(n_in, n_hidden))
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = rng.normal(0, np.sqrt(2/(n_hidden+n_out)), size=(n_hidden, n_out))
        self.b2 = np.zeros((1, n_out))
        self.lr = lr

    # ileri yayılım
    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = softmax(self.Z2)      # olasılıklar
        return self.A2

    # geri yayılım + ağırlık güncelle
    def backward(self, X, Y):
        m = X.shape[0]
        dZ2 = self.A2 - Y              
        dW2 = self.A1.T @ dZ2 / m
        db2 = dZ2.mean(axis=0, keepdims=True)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * relu_deriv(self.Z1)
        dW1 = X.T @ dZ1 / m
        db1 = dZ1.mean(axis=0, keepdims=True)

        # Gradient Descent
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1


    def fit(self, X, y, epochs=200, batch_size=32, verbose=True):
        Y = one_hot(y, self.b2.shape[1])
        history = []
        for epoch in range(1, epochs+1):
            indices = np.random.permutation(len(X))
            for start in range(0, len(X), batch_size):
                batch_idx = indices[start:start+batch_size]
                Xb, Yb = X[batch_idx], Y[batch_idx]
                self.forward(Xb)
                self.backward(Xb, Yb)

            
            probs = self.forward(X)         
            loss  = -np.log(probs[range(len(y)), y]).mean()
            history.append(loss)

            if verbose and epoch % 20 == 0:
                acc = accuracy(y, probs.argmax(axis=1))
                print(f"Epoch {epoch:4d} | Loss {loss:.4f} | Acc {acc:.2%}")
        return history

    def predict(self, X):
        return self.forward(X).argmax(axis=1)


def make_moons(n=1000, r=2.0, noise=0.2, seed=0):
    """Basit yarım ay veri kümesi (iki özellik, iki sınıf)."""
    rng = np.random.default_rng(seed)
    theta = np.pi * rng.uniform(size=n)
    x_inner = np.c_[r*np.cos(theta)   + rng.normal(0, noise, n),
                    r*np.sin(theta)   + rng.normal(0, noise, n)]
    x_outer = np.c_[r*np.cos(theta)+r + rng.normal(0, noise, n),
                   -r*np.sin(theta)+r*np.sin(np.pi) + rng.normal(0, noise, n)]
    X = np.vstack([x_inner, x_outer])
    y = np.hstack([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    perm = rng.permutation(len(X))
    return X[perm], y[perm]

X, y = make_moons(n=1000)

# train-test bölmesi
split = int(len(X)*0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


# Modeli Eğitme
model = MLP(n_in=2, n_hidden=8, n_out=2, lr=0.05)
loss_hist = model.fit(X_train, y_train, epochs=400, batch_size=64)

# sonuçlar
y_pred = model.predict(X_test)
test_acc = accuracy(y_test, y_pred)
print(f"\nTest Accuracy: {test_acc:.2%}")

fig, ax = plt.subplots(1, 2, figsize=(12,4))

ax[0].plot(loss_hist)
ax[0].set_title("Kayıp (Loss) - Epoch")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")

cm = pd.crosstab(y_test, y_pred, rownames=['Gerçek'], colnames=['Tahmin'])
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax[1])
ax[1].set_title("Karmaşıklık Matrisi")

plt.tight_layout()
plt.show()
