import dlib
import cv2
import numpy as np
import torch
import os
import random
from scipy.spatial import distance

# 加载 dlib 的人脸检测器和 landmarks 预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # 下载该文件并提供路径


def load_random_image(image_paths):
    """
    随机选择并加载一张图像。
    """
    image_path = random.choice(image_paths)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")
    return image


def get_face_landmarks(image):
    """
    检测人脸并获取 landmarks。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        raise ValueError("No face detected in the image.")

    # 选择第一个检测到的人脸
    landmarks = predictor(gray, faces[0])
    return landmarks


def landmarks_to_array(landmarks):
    """
    将 landmarks 对象转换为 numpy 数组形式。
    """
    coords = np.array([(point.x, point.y) for point in landmarks.parts()])
    return coords


def normalize_landmarks(coords):
    """
    对 landmarks 坐标进行归一化。
    """
    min_val = np.min(coords, axis=0)
    max_val = np.max(coords, axis=0)
    return (coords - min_val) / (max_val - min_val)


def compute_feature_vector(landmarks):
    """
    根据 landmarks 计算一维面部特征向量。
    使用 landmarks 之间的欧氏距离作为特征。
    """
    coords = landmarks_to_array(landmarks)
    normalized_coords = normalize_landmarks(coords)

    num_points = normalized_coords.shape[0]
    feature_vector = []

    for i in range(num_points):
        for j in range(i + 1, num_points):
            dist = distance.euclidean(normalized_coords[i], normalized_coords[j])
            feature_vector.append(dist)

    return np.array(feature_vector)


def create_feature_tensor(image_paths, batch_size):
    """
    根据图像路径列表和 batch_size 创建面部特征张量。
    """
    feature_list = []

    for _ in range(batch_size):
        # 加载随机图像
        image = load_random_image(image_paths)

        # 获取 landmarks,归一化
        landmarks = get_face_landmarks(image)

        # 计算面部特征向量
        feature_vector = compute_feature_vector(landmarks)

        # 添加到 feature 列表中
        feature_list.append(feature_vector)

    # 将 feature_list 转化为 PyTorch tensor
    feature_tensor = torch.tensor(feature_list, dtype=torch.float32)
    return feature_tensor


def average_pooling(feature_tensor, target_dim=68):
    """
    使用平均池化将高维特征降维至目标维度。

    参数:
    - feature_tensor: 原始特征张量，形状为 (1, feature_dim)
    - target_dim: 降维后的目标维度

    返回:
    - 降维后的 PyTorch 张量，形状为 (1, target_dim)
    """
    feature_array = feature_tensor.numpy()
    _, feature_dim = feature_array.shape

    # 计算每个池化块的大小
    pool_size = feature_dim // target_dim
    pooled_features = []

    for i in range(target_dim):
        start = i * pool_size
        end = start + pool_size
        # 对块内特征求平均
        pooled_features.append(np.mean(feature_array[0, start:end]))

    # 转换为 PyTorch tensor
    reduced_feature_tensor = torch.tensor([pooled_features], dtype=torch.float32)
    return reduced_feature_tensor


def extract_landmarks(b, path):
    # 读取图像
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.jpg', '.png'))]

    # 获取特征 tensor
    feature_tensor = create_feature_tensor(image_paths, b)
    reduced_feature_tensor = average_pooling(feature_tensor, target_dim=68)
    return reduced_feature_tensor
