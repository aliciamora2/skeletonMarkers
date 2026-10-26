# skeletonMarkers
This repository contains the implementation accompanying the research paper: **"A ROS-Integrated Skeleton Detection Framework Enabling Human-Centered Robotic Applications"** (ICARSC 2024).

---

## 📁 Repository Structure

The project is divided into two main folders:

1. **`robot_dance/`**
   * A **ROS package** that utilizes MediaPipe to detect a person's skeleton in real time and publish it as landmarks for visualization in RViz.

2. **`skeletonAnalysis/`**
   * Contains Python scripts for offline skeleton extraction and kinematics analysis, complete with sample data:
     * `skeleton.py`: Extracts the 3D skeleton from an image.
     * `IK.py`: Computes the joint angles of the extracted skeleton using Inverse Kinematics.

---

## 📄 Citation

If you use this code or work in your research, projects, or teaching, please cite it as:

```bibtex
@inproceedings{mora2024ros,
  title        = {A ROS-Integrated Skeleton Detection Framework Enabling Human-Centered Robotic Applications},
  author       = {Mora, Alicia and Ozeki, Tomoe and Mouri, Tetsuya and Barber, Ramon},
  booktitle    = {2024 IEEE International Conference on Autonomous Robot Systems and Competitions (ICARSC)},
  pages        = {3--8},
  year         = {2024},
  organization = {IEEE}
}
