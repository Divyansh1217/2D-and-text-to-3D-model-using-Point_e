# JustIT - 3D Model Generator with Streamlit

This Streamlit app generates 3D models from text prompts using OpenAI's Point-E and CLIP models.

## Features
- Input a text prompt to generate 3D point cloud models.
- Powered by OpenAI's `point-e` and `clip` for rendering.

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/justit.git
cd justit
# JustIT - 3D Model Generator with Streamlit

This Streamlit app generates 3D models from text prompts using OpenAI's Point-E and CLIP models.

## Features
- Input a text prompt to generate 3D point cloud models.
- Powered by OpenAI's `point-e` and `clip` for rendering.

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/justit.git
cd justit


# Point Cloud to Mesh using SDF and Marching Cubes

This project loads a 3D point cloud and converts it into a mesh using a Signed Distance Function (SDF) model and the Marching Cubes algorithm. It outputs the reconstructed mesh as a `.ply` file that can be visualized in 3D modeling tools like Blender or MeshLab.

---

## 📦 Features

- Loads point cloud data from `.npz` files
- Visualizes point cloud with color
- Uses a pretrained SDF model (`ViT-L-14.pt`) to infer a surface
- Applies Marching Cubes to generate a high-resolution mesh
- Exports the mesh in `.ply` format

---