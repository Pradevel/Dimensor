import tkinter as tk
from tkinter import filedialog, messagebox
import torch
import torchvision.transforms as T
import open3d as o3d
import cv2
import numpy as np
from PIL import Image, ImageTk

class DepthEstimationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Depth Estimation App")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        self.midas.to(self.device).eval()

        self.transform = T.Compose([
            T.Resize(384),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.image_path_var = tk.StringVar()
        tk.Label(root, text="Image Path:").pack(padx=10, pady=5)
        self.image_path_entry = tk.Entry(root, textvariable=self.image_path_var, width=50)
        self.image_path_entry.pack(padx=10, pady=5)

        self.load_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_button.pack(padx=10, pady=5)

        self.process_button = tk.Button(root, text="Process Image", command=self.process_image)
        self.process_button.pack(padx=10, pady=5)

        self.image_label = tk.Label(root, text="No image loaded")
        self.image_label.pack(padx=10, pady=5)

    def load_image(self):
        image_path = self.image_path_var.get()
        if image_path:
            try:
                img = Image.open(image_path)
                img = img.resize((300, 300))
                img_tk = ImageTk.PhotoImage(img)
                self.image_label.config(image=img_tk, text="")
                self.image_label.image = img_tk
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")
        else:
            messagebox.showwarning("Warning", "No image path provided")

    def preprocess_image(self):
        img = cv2.imread(self.image_path_var.get())
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_size = img_rgb.shape[:2]
        img_resized = cv2.resize(img_rgb, (384, 384))
        img_pil = Image.fromarray(img_resized)
        input_img = self.transform(img_pil).unsqueeze(0).to(self.device)
        return input_img, original_size, img_rgb

    def generate_depth_map(self, input_img):
        with torch.no_grad():
            depth_map = self.midas(input_img).squeeze().cpu().numpy()
        return depth_map

    def create_point_cloud(self, depth_map, original_size, img_rgb):
        depth_map = cv2.resize(depth_map, (original_size[1], original_size[0]))
        depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))

        h, w = depth_map.shape
        fx, fy = w, h
        cx, cy = w // 2, h // 2

        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x = (x.flatten() - cx) * depth_map.flatten() / fx
        y = -(y.flatten() - cy) * depth_map.flatten() / fy
        z = depth_map.flatten()

        points = np.vstack((x, y, z)).T
        colors = img_rgb.reshape(-1, 3) / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    def process_image(self):
        image_path = self.image_path_var.get()
        if not image_path:
            messagebox.showwarning("Warning", "No image path provided")
            return

        try:
            input_img, original_size, img_rgb = self.preprocess_image()
            depth_map = self.generate_depth_map(input_img)
            pcd = self.create_point_cloud(depth_map, original_size, img_rgb)

            # Visualize the point cloud
            o3d.visualization.draw_geometries([pcd])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DepthEstimationApp(root)
    root.mainloop()