# Gradient Descent 2D Visualization

A **Python-based interactive visualization** of gradient descent on the 2D function \( y = x^2 \), designed to intuitively demonstrate how gradient-based optimization works.

---
![2025-10-24 17-59-25](https://github.com/user-attachments/assets/1aaf4e32-b38f-4cf6-a98a-419f125b584a)

## 🖼 Project Overview

This project visualizes the **gradient descent algorithm** in 2D using Python, NumPy, SymPy, and Matplotlib.

**Key highlights:**

- Animated **ball moving along the curve** following gradient descent steps.
- **Interactive slider** to choose the starting \(x\)-position.
- Shows **descent steps** with a connecting line for clarity.

---

## 🎯 Learning Goals

- Understand how **gradient descent works** to minimize a function.
- Observe how the **starting position affects convergence**.
- Build intuition for **learning rate** and step sizes in optimization.

---

## ⚙️ Features

- **2D line plot visualization** for clear observation of the descent.
- **Animated ball and step lines** showing each update in real-time.
- **Interactive slider** to change the initial starting point.
- Implements symbolic derivatives using **SymPy** for automatic gradient computation.

---

## 💻 How to Run

```bash
# Clone the repository
git clone https://github.com/yourusername/gradient-descent-2d.git
cd gradient-descent-2d

# Install dependencies
pip install numpy sympy matplotlib

# Run the visualization
python gradient_descent_2d.py
