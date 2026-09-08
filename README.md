# numerical methods resolution and visualization

Interactive Python applications for exploring numerical methods through graphical interfaces and visual outputs.

The project contains two tools: a root-finding visualizer and a two-dimensional linear dynamical-system solver.

## Root-Finding Visualizer

The root-finding application implements and compares several numerical methods:

- Bisection
- Secant method
- Newton's method
- Finite-difference quasi-Newton method
- Müller's method
- Fixed-point iteration
- Random search

The user can define a function, select a numerical method and tolerance, and visualize both the estimated root and the sequence of approximations.

![Root-Finding Visualizer](assets/root_finding_demo.png)

## Linear Dynamical System Visualizer

The second application studies systems of the form

\[
\begin{aligned}
y_1'(t) &= a_{11}y_1(t) + a_{12}y_2(t) + b_1, \\
y_2'(t) &= a_{21}y_1(t) + a_{22}y_2(t) + b_2.
\end{aligned}
\]

It combines numerical and symbolic analysis by providing:

- Numerical integration with SciPy
- Symbolic solutions with SymPy
- Stationary-state calculation
- Phase portraits
- Time-series trajectories
- Animated solutions
- Numerical result tables

![Linear Dynamical System Visualizer](assets/linear_system_demo.png)

## Run the Applications

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

Root-finding application:

```bash
python apps/root_finding_visualizer.py
```

Linear dynamical-system application:

```bash
python apps/linear_dynamical_system_visualizer.py
```

## Technologies

Python, NumPy, SciPy, SymPy, Matplotlib, PySide6, and Tkinter.

## Purpose

This project was developed to explore numerical algorithms interactively and connect their mathematical formulation with numerical and graphical representations.
