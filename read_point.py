import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

# 加载点云文件 (假设点云文件为 point_cloud.ply)
pcd = o3d.io.read_point_cloud("/media/sxm/Data/smh/GaussianTalker/output/obama_jaw_without_mouth_mask/point_cloud/iteration_10000/point_cloud.ply")

# 将点云转换为 numpy 数组
points = np.asarray(pcd.points)

# 可视化原始点云
o3d.visualization.draw_geometries([pcd], window_name="Point Cloud")

# 计算点云的中心和协方差矩阵
mean = np.mean(points, axis=0)
covariance = np.cov(points.T)

# 创建 3D 网格，定义范围
x = np.linspace(mean[0] - 5, mean[0] + 5, 100)
y = np.linspace(mean[1] - 5, mean[1] + 5, 100)
z = np.linspace(mean[2] - 5, mean[2] + 5, 100)

# 创建网格坐标
X, Y, Z = np.meshgrid(x, y, z)

# 将网格坐标拉平
grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

# 计算每个网格点的高斯分布密度
rv = multivariate_normal(mean, covariance)
gaussian_density = rv.pdf(grid_points)

# 将密度转换为适合可视化的值 (例如，可以取对数，或限制范围)
gaussian_density = gaussian_density.reshape(X.shape)

# 可视化高斯分布
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制3D曲面
ax.plot_surface(X, Y, gaussian_density, cmap='viridis', edgecolor='none', alpha=0.7)

# 设置标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Density')
ax.set_title('3D Gaussian Distribution')

# 显示高斯分布图
plt.show()

# 如果需要将高斯分布与点云结合显示，可以在点云周围生成一个透明的高斯分布区域
# 使用open3d的几何图形，如体素网格（voxel grid）等
