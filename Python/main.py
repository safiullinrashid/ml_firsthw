import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Функция для подсчета расстояния между двумя точками
def calculate_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Загрузка датасета Iris
iris = load_iris()
data = iris.data

# Поиск оптимального числа кластеров с использованием библиотеки sklearn
silhouette_scores = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data)
    silhouette_avg = silhouette_score(data, cluster_labels)
    silhouette_scores.append(silhouette_avg)

optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2

# Собственная реализация алгоритма K-means
def custom_kmeans(data, n_clusters, max_iterations=100):
    n_samples, n_features = data.shape
    centroids = data[np.random.choice(n_samples, n_clusters, replace=False)]
    cluster_labels = np.zeros(n_samples)

    for _ in range(max_iterations):
        distances = np.array([np.linalg.norm(data - centroid, axis=1) for centroid in centroids])
        new_cluster_labels = np.argmin(distances, axis=0)

        if np.array_equal(cluster_labels, new_cluster_labels):
            break

        cluster_labels = new_cluster_labels.copy()

        # Визуализация шага
        plt.figure()
        colors = ['red', 'green', 'blue', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta', 'yellow']
        for cluster in range(n_clusters):
            plt.scatter(data[cluster_labels == cluster][:, 0], data[cluster_labels == cluster][:, 1], label=f'Кластер {cluster+1}', color=colors[cluster])
            plt.scatter(centroids[cluster][0], centroids[cluster][1], color='black', marker='x', s=100, label='Центроид')
        plt.title(f'Итерация: {_+1}')
        plt.legend()
        plt.show()

        # Обновление центроидов
        for i in range(n_clusters):
            centroids[i] = data[cluster_labels == i].mean(axis=0)

    return centroids, cluster_labels

# Применение алгоритма к датасету Iris с оптимальным числом кластеров
centroids, cluster_labels = custom_kmeans(data[:, 2:4], optimal_clusters)