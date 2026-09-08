import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import sympy as sp

# Global variables to store the current animation and its pause state
ani = None
paused = False
pause_button = None


def system(t, y, a11, a12, a21, a22, b1, b2):
    """
    Définit le système d'équations différentielles:
      y1'(t) = a11 * y1(t) + a12 * y2(t) + b1
      y2'(t) = a21 * y1(t) + a22 * y2(t) + b2
    """
    y1, y2 = y
    return [a11 * y1 + a12 * y2 + b1, a21 * y1 + a22 * y2 + b2]


def validate_float(entry_widget, name):
    """
    Tente de convertir le texte de entry_widget en float.
    Si la conversion échoue, affiche un message d'erreur et renvoie None.
    Sinon, renvoie la valeur float.
    """
    try:
        return float(entry_widget.get())
    except ValueError:
        messagebox.showerror("Entrée invalide", f"Veuillez entrer une valeur numérique pour l'input: {name}.")
        return None


def solve_and_plot(entries, canvas_frame, interval_slider, auto_replay, plot_type_var):

    """
    Récupère les paramètres (y compris dt) depuis l'interface, résout le système d'équations,
    et affiche une animation ainsi qu'une table de résultats (avec deux labels au-dessus de la table)
    sur le côté droit de la fenêtre.
    """
    global ani, paused, pause_button

    # Effacer le contenu précédent dans le cadre de la figure/table
    for widget in canvas_frame.winfo_children():
        widget.destroy()

    # Réinitialiser l'état de pause
    paused = False
    if pause_button is not None:
        pause_button.config(text="Pause")

    # -----------------------------
    # Validation des entrées utilisateur
    # -----------------------------
    a11 = validate_float(entries["a11"], "a11")
    if a11 is None: return
    a12 = validate_float(entries["a12"], "a12")
    if a12 is None: return
    a21 = validate_float(entries["a21"], "a21")
    if a21 is None: return
    a22 = validate_float(entries["a22"], "a22")
    if a22 is None: return
    b1 = validate_float(entries["b1"], "b1")
    if b1 is None: return
    b2 = validate_float(entries["b2"], "b2")
    if b2 is None: return
    y1_0 = validate_float(entries["y1(0)"], "y1(0)")
    if y1_0 is None: return
    y2_0 = validate_float(entries["y2(0)"], "y2(0)")
    if y2_0 is None: return
    t0 = validate_float(entries["t_début"], "t_début")
    if t0 is None: return
    t1 = validate_float(entries["t_fin"], "t_fin")
    if t1 is None: return
    dt = validate_float(entries["dt"], "dt")
    if dt is None: return

    interval_val = float(interval_slider.get())  # intervalle en ms pour l'animation

    # Construire le vecteur temps
    num_points = int((t1 - t0) / dt) + 1
    t_eval = np.linspace(t0, t1, num_points)

    # Résoudre le système d'équations différentielles numériquement
    sol = solve_ivp(system, (t0, t1), [y1_0, y2_0],
                    args=(a11, a12, a21, a22, b1, b2),
                    t_eval=t_eval)
    y1_sol, y2_sol = sol.y[0], sol.y[1]

    # -----------------------------
    # Création de la figure Matplotlib
    # -----------------------------
    fig, ax = plt.subplots(figsize=(6, 6))

    if plot_type_var.get() == "phase":
        ax.set_xlabel("y1")
        ax.set_ylabel("y2")
        ax.set_title("Diagramme de phase : y1 vs y2")
        ax.plot(y1_sol, y2_sol, color='gray', alpha=0.3, label='Trajectoire')
        ax.plot(y1_sol[0], y2_sol[0], 'go', label='Début')
        ax.plot(y1_sol[-1], y2_sol[-1], 'ro', label='Fin')
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
        line, = ax.plot([], [], 'b-', lw=2, label='Trajet animé')
        point, = ax.plot([], [], 'ro', label='Mobile')
        padding_x = 0.1 * (y1_sol.max() - y1_sol.min())
        padding_y = 0.1 * (y2_sol.max() - y2_sol.min())
        ax.set_xlim(y1_sol.min() - padding_x, y1_sol.max() + padding_x)
        ax.set_ylim(y2_sol.min() - padding_y, y2_sol.max() + padding_y)
        ax.legend()

        def init():
            line.set_data([], [])
            point.set_data([], [])
            return line, point

        def update(frame):
            line.set_data(y1_sol[:frame + 1], y2_sol[:frame + 1])
            point.set_data([y1_sol[frame]], [y2_sol[frame]])
            return line, point

    else:  # Courbes y1(t) et y2(t)
        ax.set_xlabel("t")
        ax.set_ylabel("y1(t), y2(t)")
        ax.set_title("Évolution de y1(t) et y2(t)")
        ax.plot(t_eval, y1_sol, color='blue', alpha=0.3, label='y1(t)')
        ax.plot(t_eval, y2_sol, color='green', alpha=0.3, label='y2(t)')
        line1, = ax.plot([], [], 'b-', label='y1(t)')
        line2, = ax.plot([], [], 'g-', label='y2(t)')
        point1, = ax.plot([], [], 'bo')
        point2, = ax.plot([], [], 'go')
        ax.set_xlim(t_eval[0], t_eval[-1])
        y_combined = np.concatenate([y1_sol, y2_sol])
        padding_y = 0.1 * (y_combined.max() - y_combined.min())
        ax.set_ylim(y_combined.min() - padding_y, y_combined.max() + padding_y)
        ax.legend()

        def init():
            line1.set_data([], [])
            line2.set_data([], [])
            point1.set_data([], [])
            point2.set_data([], [])
            return line1, line2, point1, point2

        def update(frame):
            line1.set_data(t_eval[:frame + 1], y1_sol[:frame + 1])
            line2.set_data(t_eval[:frame + 1], y2_sol[:frame + 1])
            point1.set_data([t_eval[frame]], [y1_sol[frame]])
            point2.set_data([t_eval[frame]], [y2_sol[frame]])
            return line1, line2, point1, point2

    # Créer l'animation (le paramètre repeat est contrôlé par le switch "Rejouer automatiquement")
    ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init,
                        blit=True, interval=interval_val, repeat=auto_replay.get())

    # -----------------------------------------------------
    # Disposition : Figure à gauche, Équations + Table à droite
    # -----------------------------------------------------
    plot_table_frame = tk.Frame(canvas_frame)
    plot_table_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Cadre de gauche pour la figure (canvas + barre d'outils)
    canvas_frame_left = tk.Frame(plot_table_frame)
    canvas_frame_left.grid(row=0, column=0, sticky="nsew")

    # Cadre de droite pour "Équations (en haut) + Table (en bas)"
    right_frame = tk.Frame(plot_table_frame)
    right_frame.grid(row=0, column=1, sticky="nsew")

    plot_table_frame.columnconfigure(0, weight=1)
    plot_table_frame.columnconfigure(1, weight=1)
    plot_table_frame.rowconfigure(0, weight=1)

    # -----------------------------
    # Intégrer la figure Matplotlib
    # -----------------------------
    canvas = FigureCanvasTkAgg(fig, master=canvas_frame_left)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    toolbar = NavigationToolbar2Tk(canvas, canvas_frame_left)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # -----------------------------------------------------
    # 1) Cadre pour afficher les équations
    # -----------------------------------------------------
    equations_frame = tk.Frame(right_frame)
    equations_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

    # Étiquette du système général
    general_eq_text = (
        "Système d'équations général:\n"
        "   y1'(t) = a11 * y1(t) + a12 * y2(t) + b1,\n"
        "   y2'(t) = a21 * y1(t) + a22 * y2(t) + b2\n"
        ""
    )
    eq_label = tk.Label(equations_frame, text=general_eq_text, justify="left",
                        font=("Arial", 10, "bold"))
    eq_label.pack(side=tk.TOP, anchor="w")

    # Étiquette dynamique avec les valeurs actuelles
    current_eq_text = (
        "Système actuel:\n"
        f"   y1'(t) = {a11} * y1(t) + {a12} * y2(t) + {b1},\n"
        f"   y2'(t) = {a21} * y1(t) + {a22} * y2(t) + {b2}\n"
        ""
    )
    current_eq_label = tk.Label(equations_frame, text=current_eq_text, justify="left",
                                font=("Arial", 10))
    current_eq_label.pack(side=tk.TOP, anchor="w")

    # -------------------------------
    # Résolution symbolique du système
    # -------------------------------
    # Définition des symboles et fonctions
    t_sym = sp.symbols('t')
    y1_sym = sp.Function('y1')(t_sym)
    y2_sym = sp.Function('y2')(t_sym)
    # Définition des équations avec les valeurs récupérées
    eq1_sym = sp.Eq(sp.diff(y1_sym, t_sym), a11 * y1_sym + a12 * y2_sym + b1)
    eq2_sym = sp.Eq(sp.diff(y2_sym, t_sym), a21 * y1_sym + a22 * y2_sym + b2)
    # Résolution symbolique
    sol_sym = sp.dsolve([eq1_sym, eq2_sym])
    # Conversion de la solution en une chaîne de caractères bien formatée
    # Résumé de la solution symbolique sur deux lignes

    def round_constants(expr, decimals=3):
        """
        Parcours récursivement l'expression et arrondit tous les nombres flottants à `decimals`.
        """
        if expr.is_Number:
            return sp.Float(expr.evalf(decimals))
        elif expr.is_Atom:
            return expr
        else:
            return expr.func(*[round_constants(arg, decimals) for arg in expr.args])

    # Appliquer l'arrondi récursif à chaque équation
    y1_rounded = round_constants(sol_sym[0].rhs, decimals=3)
    y2_rounded = round_constants(sol_sym[1].rhs, decimals=3)

    def truncate_expression(expr, max_chars=200):
        """
        Truncates a SymPy expression to a string of at most `max_chars` characters.
        Ensures the start of the expression is visible.
        """
        expr_str = str(expr)
        if len(expr_str) <= max_chars:
            return expr_str
        else:
            return expr_str[:max_chars] + " ..."

    # Appliquer le tronquage à chaque solution symbolique
    y1_str = truncate_expression(y1_rounded)
    y2_str = truncate_expression(y2_rounded)

    # Formater le texte final
    sol_str = f"y1(t) = {y1_str}\n-----------------------------\ny2(t) = {y2_str}"
    resolved_eq_text = f"Système résolu:\n{sol_str}"

    resolved_eq_textbox = tk.Text(equations_frame, height=6, wrap="word", font=("Arial", 10))
    resolved_eq_textbox.insert("1.0", resolved_eq_text)
    resolved_eq_textbox.config(state="disabled")  # Make it read-only
    resolved_eq_textbox.pack(side=tk.TOP, fill="x", padx=2, pady=2)
    # -----------------------------------------------------
    # État stationnaire
    # -----------------------------------------------------
    try:
        A = np.array([[a11, a12], [a21, a22]])
        b_vec = np.array([-b1, -b2])
        y_star = np.linalg.solve(A, b_vec)
        steady_state_text = f"État stationnaire:\n   y1* = {y_star[0]:.3f}, y2* = {y_star[1]:.3f}"
    except np.linalg.LinAlgError:
        steady_state_text = "État stationnaire: aucun (matrice singulière)"

    steady_label = tk.Label(equations_frame, text=steady_state_text,
                            justify="left", font=("Arial", 10, "italic"), fg="darkblue")
    steady_label.pack(side=tk.TOP, anchor="w", pady=(5, 0))

    # -----------------------------------------------------
    # 2) Cadre pour la table des résultats (sous les équations)
    # -----------------------------------------------------
    table_frame = tk.Frame(right_frame)
    table_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    scrollbar = tk.Scrollbar(table_frame, orient="vertical")
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    table = ttk.Treeview(table_frame, columns=("Temps", "y1", "y2"), show="headings",
                         yscrollcommand=scrollbar.set)
    table.heading("Temps", text="Temps")
    table.heading("y1", text="y1")
    table.heading("y2", text="y2")
    table.column("Temps", width=80, anchor="center")
    table.column("y1", width=80, anchor="center")
    table.column("y2", width=80, anchor="center")
    table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.config(command=table.yview)

    # Insertion des données (avec 3 décimales)
    for i in range(len(t_eval)):
        table.insert("", "end", values=(f"{t_eval[i]:.3f}", f"{y1_sol[i]:.3f}", f"{y2_sol[i]:.3f}"))


def toggle_pause():
    """
    Bascule l'état pause/reprise de l'animation courante.
    """
    global ani, paused, pause_button
    if ani is None:
        return
    if paused:
        ani.event_source.start()
        paused = False
        pause_button.config(text="Pause")
    else:
        ani.event_source.stop()
        paused = True
        pause_button.config(text="Reprendre")


def main():
    """
    Construit une interface graphique Tkinter pour saisir les paramètres (y compris dt),
    la plage de temps, et l'intervalle d'animation (via un slider). Intègre la figure animée
    Matplotlib et la table de résultats (avec deux étiquettes d'équations en haut de la table),
    et ajoute également un bouton Quitter et un switch pour rejouer automatiquement.
    """
    global pause_button
    root = tk.Tk()
    root.title("2D Linear Dynamical System Visualizer")

    # Cadre de gauche pour les widgets d'input; cadre de droite pour la figure/équations/table
    input_frame = tk.Frame(root)
    input_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10)
    plot_frame = tk.Frame(root)
    plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Dictionnaire pour les entrées
    entries = {}
    # Liste de paires (label, valeur par défaut)
    parameters = [
        ("a11", "-0.1"), ("a12", "0.2"), ("a21", "-0.3"), ("a22", "-0.4"),
        ("b1", "1.0"), ("b2", "-1.0"), ("y1(0)", "1.0"), ("y2(0)", "0.5"),
        ("t_début", "0"), ("t_fin", "10"), ("dt", "0.02")
    ]
    for row, (label_text, default) in enumerate(parameters):
        tk.Label(input_frame, text=label_text + ":").grid(row=row, column=0, sticky="e", pady=2)
        entry = tk.Entry(input_frame)
        entry.grid(row=row, column=1, pady=2)
        entry.insert(0, default)
        entries[label_text] = entry

    # Slider pour l'intervalle d'animation (en ms) de 0 à 100 avec pas de 5
    slider_interval = tk.Scale(input_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                               resolution=5, label="Intervalle (ms)")
    slider_interval.set(5)  # valeur par défaut
    slider_interval.grid(row=len(parameters), column=0, columnspan=2, pady=10)

    # Switch pour "Rejouer automatiquement" (True = rejouer)
    auto_replay = tk.BooleanVar(value=False)
    replay_switch = ttk.Checkbutton(input_frame, text="Rejouer automatiquement", variable=auto_replay)
    replay_switch.grid(row=len(parameters) + 1, column=0, columnspan=2, pady=5)

    # Choix du type de graphique (diagramme de phase ou courbes y1/y2)
    plot_type_var = tk.StringVar(value="phase")
    plot_type_label = tk.Label(input_frame, text="Type de graphique :")
    plot_type_label.grid(row=len(parameters) + 2, column=0, pady=5)
    plot_type_menu = ttk.Combobox(input_frame, textvariable=plot_type_var,
                                  values=["phase", "courbes y1(t), y2(t)"], state="readonly")
    plot_type_menu.grid(row=len(parameters) + 2, column=1, pady=5)

    # Bouton pour déclencher la résolution et l'animation
    solve_button = ttk.Button(input_frame, text="Résoudre et animer",
                              command=lambda: solve_and_plot(entries, plot_frame, slider_interval, auto_replay, plot_type_var))

    solve_button.grid(row=len(parameters) + 2, column=0, columnspan=2, pady=10)

    # Bouton Pause/Reprendre
    pause_button = ttk.Button(input_frame, text="Pause", command=toggle_pause)
    pause_button.grid(row=len(parameters) + 3, column=0, columnspan=2, pady=10)

    # Bouton Quitter
    quit_button = ttk.Button(input_frame, text="Quitter", command=root.destroy)
    quit_button.grid(row=len(parameters) + 4, column=0, columnspan=2, pady=10)

    root.mainloop()


if __name__ == "__main__":
    main()
