# basic matplotlib
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8,6), dpi=80)
# subplot 1
sp1 = plt.subplot(2, 2, 1)
X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
C,S = np.cos(X), np.sin(X)*2
#xlim(-4,4)
plt.plot(X,C)
#plt.xticks(-2,2,5, endpoint=True)

# subplot 2
sp2 = plt.subplot(2, 2, 2)
plt.plot(X,S,color="red")
plt.xticks(np.linspace(-4,4,9, endpoint=True))



plt.show()