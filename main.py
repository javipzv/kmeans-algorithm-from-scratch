import numpy as np
import matplotlib.pyplot as plt
from kmeans_algorithm import kmeans 

n = 200
s = 0.4

x_1 = np.random.normal(loc=1, scale=s, size=n)
y_1 = np.random.normal(loc=1, scale=s, size=n)

x_2 = np.random.normal(loc=3, scale=s, size=n)
y_2 = np.random.normal(loc=1, scale=s, size=n)

x_3 = np.random.normal(loc=2, scale=s, size=n)
y_3 = np.random.normal(loc=3, scale=s, size=n)

x_4 = np.random.normal(loc=3, scale=s, size=n)
y_4 = np.random.normal(loc=5, scale=s, size=n)

x_5 = np.random.normal(loc=4, scale=s, size=n)
y_5 = np.random.normal(loc=3, scale=s, size=n)

data = np.array([np.concatenate((x_1, x_2, x_3, x_4, x_5)), np.concatenate((y_1, y_2, y_3, y_4, y_5))]).T

labels, centroids, model_inertia = kmeans(data=data, k=5, initialization='kmeans++', n_init=5, max_iter=10)
print(labels[:, 2])

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

ax1 = axs[0]
ax2 = axs[1]

ax1.grid(True)
ax1.scatter(data[:, 0], data[:, 1], color='blue')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Original data')

ax2.grid(True)
ax2.scatter(labels[:, 0], labels[:, 1], c=labels[:, 2])
ax2.scatter(np.array(list(centroids.values()))[:, 0], np.array(list(centroids.values()))[:, 1], s=100, c='red')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Clustered data with inertia: ' + str(np.round(model_inertia, 2)))
plt.show()