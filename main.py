# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def k_means(data, k, max_iter=100, tol=1e-4):
	"""
	Implementation of K-means
	:param:
	- data: 数据集 (numpy array)
	- k: 聚类数量
	- max_iter: 最大迭代次数
	- tol: 收敛阈值 (默认 1e-4)
	:return
	- clusters: 每个样本的聚类标签
	- centroids: 聚类中心点
	"""
	# 随机初始化 k 个聚类中心
	np.random.seed(42)  # 固定随机种子，方便复现
	centroids = data[np.random.choice(data.shape[0], k, replace=False)]

	for i in range(max_iter):
		# 计算每个点到聚类中心的欧几里得距离，并归类
		distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
		clusters = np.argmin(distances, axis=1)

		# 计算新的聚类中心
		new_centroids = np.array([data[clusters == j].mean(axis=0) for j in range(k)])

		# 检查是否收敛
		if np.linalg.norm(new_centroids - centroids) < tol:
			break
		centroids = new_centroids

	return clusters, centroids

if __name__ == "__main__":
	# 读取csv数据
	data = pd.read_csv('4.0.csv', encoding='utf-8')

	# 查看数据
	print(data.head())

	# 提取特征：密度和含糖率
	features = data[['density', 'sugercontent']].values  # 提取这两列数据

	# 假设 k = 3
	k = 3
	clusters, centroids = k_means(features, k)

	# 可视化
	import matplotlib

	matplotlib.use('TkAgg')  # 设置为 TkAgg 后端，避免与图形界面冲突
	plt.figure(figsize=(8, 6))
	for i in range(k):
		plt.scatter(features[clusters == i, 0], features[clusters == i, 1], label=f'Cluster {i + 1}')

	plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
	plt.title("K-Means Clustering")
	plt.xlabel("density")
	plt.ylabel("sugercontent")
	plt.legend()
	plt.grid()
	plt.show()