import numpy as np
import sys
import time
import matplotlib.pyplot as plt
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
import sympy as sp
import math
import numdifftools as nd
import random


def bisection(f, min_val, max_val, epsilon=1e-6):
    approximations = []

    f_min = f(min_val)
    f_max = f(max_val)

    if f_min == 0:
        return min_val, approximations
    if f_max == 0:
        return max_val, approximations

    if f_min * f_max > 0:
        raise Exception(f"Les limites ne contiennent pas la racine: f({min_val}) = {f_min}, f({max_val}) = {f_max}")

    for i in range(1000):
        m = (min_val + max_val) / 2.0
        f_m = f(m)
        approximations.append(m)

        if abs(f_m) < epsilon:
            return m, approximations
        if f_min * f_m > 0:
            min_val, f_min = m, f_m
        else:
            max_val = m

    return m, approximations


def secante(f, min_val, max_val, epsilon=1e-6, max_iter=100):
    approximations = [min_val, max_val]

    for _ in range(max_iter):
        f_min = f(min_val)
        f_max = f(max_val)

        if abs(f_max) < epsilon:
            return max_val, approximations

        if f_max - f_min == 0:
            raise Exception("Division par zero")

        x2 = max_val - f_max * (max_val - min_val) / (f_max - f_min)
        approximations.append(x2)

        if abs(x2 - max_val) < epsilon:
            return x2, approximations

        min_val, max_val = max_val, x2

    return max_val, approximations


def newton(f, x0, epsilon=1e-6, max_iter=100):
    approximations = [x0]

    for _ in range(max_iter):
        f_x0 = f(x0)
        df_x0 = nd.Derivative(f)(x0)

        if df_x0 == 0:
            raise Exception(f"La dérivée est zéro à x = {x0}")

        x1 = x0 - f_x0 / df_x0
        approximations.append(x1)

        if abs(x1 - x0) < epsilon:
            return x1, approximations

        x0 = x1

    raise Exception(f"La méthode n'a pas convergé après {max_iter} itérations.")

def Quasinewton(f, x0, epsilon=1e-6, max_iter=100):
    #approximation de la derive par difference fini
    def df(f, x):
        return (f(x + 1e-5) - f(x - 1e-5)) / (2 * 1e-5)

    approximations = [x0]
    # methode de newton
    for _ in range(max_iter):
        f_x0 = f(x0)
        df_x0 = df(f, x0)

        if df_x0 == 0:
            raise Exception(f"la derive est zero a x = {x0}")

        x1 = x0 - f_x0 / df_x0
        approximations.append(x1)

        if abs(x1 - x0) < epsilon:
            return x1, approximations

        x0 = x1

    raise Exception(f"La methode na pas converger apres {max_iter} iterations.")


def muller(f, x0, x2, epsilon=1e-6, max_iter=100):
    x1 = (x0 + x2) / 2
    approximations = [x1]

    for _ in range(max_iter):
        f0, f1, f2 = f(x0), f(x1), f(x2)

        # calcule du coefficant de l'equation quadratique
        h0 = x1 - x0
        h1 = x2 - x1
        delta0 = (f1 - f0) / h0
        delta1 = (f2 - f1) / h1

        a = (delta1 - delta0) / (h1 + h0)
        b = a * h1 + delta1
        c = f2

        # calcule du discriminant de l'equation quadratique
        discriminant = math.sqrt(b ** 2 - 4 * a * c)

        # retient la valeur avec le plus grand denominateur
        denom = b + discriminant if abs(b + discriminant) > abs(b - discriminant) else b - discriminant

        # regarde que le denominateur ne soit pas 0
        if denom == 0:
            return "erreur de division par zero."

        # nouvelle approximation
        dx = -2 * c / denom
        x3 = x2 + dx
        approximations.append(x3)

        # tester la convergance
        if abs(dx) < epsilon:
            return x3, approximations

        # point pour la prochaine ittération
        x0, x1, x2 = x1, x2, x3

    return "le nombre d'itteration maximum a ete atteint sans convergence"


def point_fixe(f, x0, epsilon=1e-6, max_iter=1000, lambda_factor=0.1):
    approximation = [x0]

    # f(x) = 0 vers g(x) = x
    x = sp.symbols('x')
    fx = f(x)
    gx = x - lambda_factor * fx
    g = sp.lambdify(x, gx)

    x_n = x0
    for _ in range(max_iter):
        x1 = g(x_n)
        approximation.append(x1)
        if abs(x1 - x_n) < epsilon:
            return x1, approximation
        x_n = x1
    raise ValueError("la fonction n'a pas converger malgré le nombre maximum d'itteration")


def aleatoire(f, min_val, max_val, epsilon=0.1, max_iter=100000):
    approximation = []
    for _ in range(max_iter):
        x1 = round(random.uniform(min_val, max_val), 2)
        approximation.append(x1)
        if abs(f(x1)) - 0 < epsilon:
            return x1, approximation
    raise ValueError("aucun resultat concluant en 100 000 itterations")


class SimpleWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Root-Finding Visualizer")
        self.setGeometry(100, 100, 400, 600)

        layout = QVBoxLayout()

        title_label = QLabel("Recherche de racines d'une fonction")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; font-family: Helvetica;")
        layout.addWidget(title_label)

        labels = [
            "Fonction", "Precision", "Min", "Max", "Bissection", "Secante", "Newton",
            "Quasi-Newton", "Muller", "Point Fixe", "aleatoire", "Graphique Racine",
            "Graphique Approximations", "Animation ordinateur"
        ]

        self.method_group = QButtonGroup(self)
        self.method_group.setExclusive(True)

        self.fonction_input = None
        self.precision_input = None
        self.min_input = None
        self.max_input = None
        self.method_checkboxes = {}

        for i, label_text in enumerate(labels, start=1):
            if i in [2, 5, 12]:
                layout.addSpacing(20)

            row_layout = QHBoxLayout()
            label = QLabel(label_text)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet(
                "border: 2px solid black; padding: 8px; background-color: grey; color: white; font-family: Helvetica;"
            )
            label.setFixedSize(160, 40)
            row_layout.addWidget(label)

            if i == 1:
                fx_label = QLabel("f(x) =")
                fx_label.setAlignment(Qt.AlignCenter)
                fx_label.setStyleSheet("font-family: Helvetica;")
                fx_label.setFixedSize(50, 40)
                row_layout.addWidget(fx_label)

                self.fonction_input = QLineEdit()
                self.fonction_input.setFixedSize(160, 40)
                self.fonction_input.setStyleSheet("font-family: Helvetica;")
                row_layout.addWidget(self.fonction_input)
            elif i == 2:
                self.precision_input = QLineEdit()
                self.precision_input.setFixedSize(160, 40)
                self.precision_input.setStyleSheet("font-family: Helvetica;")
                row_layout.addWidget(self.precision_input)
            elif i == 3:
                self.min_input = QLineEdit()
                self.min_input.setFixedSize(160, 40)
                self.min_input.setStyleSheet("font-family: Helvetica;")
                row_layout.addWidget(self.min_input)
            elif i == 4:
                self.max_input = QLineEdit()
                self.max_input.setFixedSize(160, 40)
                self.max_input.setStyleSheet("font-family: Helvetica;")
                row_layout.addWidget(self.max_input)
            elif 5 <= i <= 11:
                checkbox = QCheckBox()
                checkbox.setFixedSize(40, 40)
                checkbox.setStyleSheet("font-family: Helvetica;")
                self.method_group.addButton(checkbox)
                row_layout.addWidget(checkbox)
                self.method_checkboxes[label_text] = checkbox
            else:
                checkbox = QCheckBox()
                checkbox.setFixedSize(40, 40)
                checkbox.setStyleSheet("font-family: Helvetica;")
                row_layout.addWidget(checkbox)
                self.method_checkboxes[label_text] = checkbox

            layout.addLayout(row_layout)

        layout.addStretch()

        button_layout = QHBoxLayout()
        button_layout.addSpacerItem(QSpacerItem(300, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        self.button = QPushButton("Calculer")
        self.button.setFixedSize(200, 50)
        self.button.setStyleSheet("font-family: Helvetica;")
        self.button.clicked.connect(self.calculate_function)
        button_layout.addWidget(self.button)

        button_layout.addSpacerItem(QSpacerItem(300, 100, QSizePolicy.Expanding, QSizePolicy.Minimum))
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def calculate_function(self):
        try:
            func_str = self.fonction_input.text()
            global f
            f = lambda x: eval(func_str, {"x": x, "math": __import__("math"), "np": np})

            min_val = float(self.min_input.text())
            max_val = float(self.max_input.text())
            precision = float(self.precision_input.text())

            selected_method = None
            for method, checkbox in self.method_checkboxes.items():
                if checkbox.isChecked():
                    selected_method = method
                    break

            if selected_method == "Bissection":
                start_time = time.time()
                root, approximations = bisection(f, min_val, max_val, epsilon=precision)
                end_time = time.time()
                elapsed_time = end_time - start_time

                QMessageBox.information(self, "Résultat",
                                        f"La racine trouvée est: {root}\n"
                                        f"Temps de calcul: {elapsed_time:.6f} secondes\n"
                                        f"les itteration sont: {approximations}")

                if self.method_checkboxes.get("Graphique Approximations").isChecked():
                    self.show_convergence_graph(approximations)

                if self.method_checkboxes.get("Graphique Racine").isChecked():
                    self.show_function_graph(f, min_val, root, max_val)

            if selected_method == "Secante":
                start_time = time.time()
                root, approximations = secante(f, min_val, max_val, epsilon=precision)
                end_time = time.time()
                elapsed_time = end_time - start_time

                QMessageBox.information(self, "Résultat",
                                        f"La racine trouvée est: {root}\n"
                                        f"Temps de calcul: {elapsed_time:.6f} secondes\n"
                                        f"les itteration sont: {approximations}")

                if self.method_checkboxes.get("Graphique Approximations").isChecked():
                    self.show_convergence_graph(approximations)

                if self.method_checkboxes.get("Graphique Racine").isChecked():
                    self.show_function_graph(f, min_val, root, max_val)

            if selected_method == "Newton":
                start_time = time.time()
                root, approximations = newton(f, min_val, epsilon=precision)
                end_time = time.time()
                elapsed_time = end_time - start_time

                QMessageBox.information(self, "Résultat",
                                        f"La racine trouvée est: {root}\n"
                                        f"Temps de calcul: {elapsed_time:.6f} secondes\n"
                                        f"les itteration sont: {approximations}")

                if self.method_checkboxes.get("Graphique Approximations").isChecked():
                    self.show_convergence_graph(approximations)

                if self.method_checkboxes.get("Graphique Racine").isChecked():
                    self.show_function_graph(f, min_val, root)

            if selected_method == "Quasi-Newton":
                start_time = time.time()
                root, approximations = Quasinewton(f, min_val, epsilon=precision)
                end_time = time.time()
                elapsed_time = end_time - start_time

                QMessageBox.information(self, "Résultat",
                                        f"La racine trouvée est: {root}\n"
                                        f"Temps de calcul: {elapsed_time:.6f} secondes\n"
                                        f"les itteration sont: {approximations}")

                if self.method_checkboxes.get("Graphique Approximations").isChecked():
                    self.show_convergence_graph(approximations)

                if self.method_checkboxes.get("Graphique Racine").isChecked():
                    self.show_function_graph(f, min_val, root)

            if selected_method == "Muller":
                start_time = time.time()
                root, approximations = muller(f, min_val, max_val, epsilon=precision)
                end_time = time.time()
                elapsed_time = end_time - start_time

                QMessageBox.information(self, "Résultat",
                                        f"La racine trouvée est: {root}\n"
                                        f"Temps de calcul: {elapsed_time:.6f} secondes\n"
                                        f"les itteration sont: {approximations}")


                if self.method_checkboxes.get("Graphique Approximations").isChecked():
                    self.show_convergence_graph(approximations)

                if self.method_checkboxes.get("Graphique Racine").isChecked():
                    self.show_function_graph(f, min_val, root, max_val)

            if selected_method == "Point Fixe":
                start_time = time.time()
                root, approximations = point_fixe(f, min_val, epsilon=precision)
                end_time = time.time()
                elapsed_time = end_time - start_time

                QMessageBox.information(self, "Résultat",
                                        f"La racine trouvée est: {root}\n"
                                        f"Temps de calcul: {elapsed_time:.6f} secondes\n"
                                        f"les itteration sont: {approximations}")


                if self.method_checkboxes.get("Graphique Approximations").isChecked():
                    self.show_convergence_graph(approximations)

                if self.method_checkboxes.get("Graphique Racine").isChecked():
                    self.show_function_graph(f, min_val, root)

            if selected_method == "aleatoire":
                start_time = time.time()
                root, approximations = aleatoire(f, min_val, max_val, epsilon=precision)
                end_time = time.time()
                elapsed_time = end_time - start_time

                QMessageBox.information(self, "Résultat",
                                        f"La racine trouvée est: {root}\n"
                                        f"Temps de calcul: {elapsed_time:.6f} secondes\n"
                                        f"le nombre ditterations est: {len(approximations)}")


                if self.method_checkboxes.get("Graphique Approximations").isChecked():
                    self.show_convergence_graph(approximations)

                if self.method_checkboxes.get("Graphique Racine").isChecked():
                    self.show_function_graph(f, min_val, root)

        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Fonction ou paramètre invalide: {e}")

    def show_convergence_graph(self, approximations):
        plt.figure(figsize=(8, 5))
        plt.plot(range(len(approximations)), approximations, marker='o', linestyle='-', color='b', label="Approximations")
        plt.axhline(y=approximations[-1], color='r', linestyle='--', label="Racine Approximée")
        plt.xlabel("Itérations")
        plt.ylabel("Valeur de x")
        plt.title("Convergence de la méthode")
        plt.legend()
        plt.grid(True)
        plt.show()

    def show_function_graph(self, f, min_val, root, max_val=10 ):
        x_vals = np.linspace(min_val - 1, max_val + 1, 400)
        y_vals = np.array([f(x) for x in x_vals])

        plt.figure(figsize=(8, 5))
        plt.plot(x_vals, y_vals, label="f(x)", color='blue')
        plt.axhline(0, color='black', linestyle='--')
        plt.axvline(0, color='black', linestyle='--')
        plt.axvline(root, color='red', linestyle='--', label=f"Racine ≈ {root:.6f}")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Graphique de la fonction étudiée")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SimpleWindow()
    window.show()
    sys.exit(app.exec())


