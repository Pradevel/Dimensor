import cv2
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def estimate_depth(image_path):
    model_type = "DPT_Large"
    midas = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True)
    midas.eval()
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").default_transform
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_img = transform(img).to('cpu')

    with torch.no_grad():
        prediction = midas(input_img)

    depth_map = prediction.squeeze().cpu().numpy()
    return depth_map


def create_point_cloud(depth_map):
    rows, cols = depth_map.shape
    point_cloud = []

    for i in range(rows):
        for j in range(cols):
            z = depth_map[i, j]
            x = (j - cols / 2) / cols * z
            y = (i - rows / 2) / rows * z
            point_cloud.append([x, y, z])

    point_cloud = np.array(point_cloud)

    if point_cloud.shape[1] != 3:
        raise ValueError(f"Expected point_cloud with 3 columns, got shape: {point_cloud.shape}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    return pcd


def visualize_point_cloud(pcd):
    o3d.visualization.draw_geometries([pcd])


def visualize_depth_map(depth_map):
    normalized_depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    normalized_depth_map = normalized_depth_map.astype(np.uint8)
    plt.imshow(normalized_depth_map, cmap='gray')
    plt.colorbar()
    plt.title('Depth Map')
    plt.show()


if __name__ == "__main__":
    image_path = "img.png"  # Replace with your image path
    depth_map = estimate_depth(image_path)
    visualize_depth_map(depth_map)
    point_cloud = create_point_cloud(depth_map)
    visualize_point_cloud(point_cloud)