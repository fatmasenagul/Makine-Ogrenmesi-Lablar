## burada eig fonksiyonunu kullanarak hocanın attığı şeyde yapılan şeyi yapıcaz ve ikisini birbiri ile kıyaslayacağız

import numpy as np
from numpy import linalg

A = np.array([[6, 1, -1],
         [0, 7, 0],
         [3, -1, 2]])

eigenvalues, eigenvektors = linalg.eig(A)

print(eigenvalues)
