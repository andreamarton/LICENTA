import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import customtkinter
import pandas as pd
import importlib
from threading import Thread
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import seaborn as sns

window = tk.Tk()
window.title('Detecția stărilor de avarie')
window.geometry('1200x750')
window.columnconfigure(0, weight=1)
window.rowconfigure(1, weight=1)

fig, ax = plt.subplots()
data = None
canvas = None

#titlu
title_frame = tk.Frame(window)
title_frame.grid(row=0, column=0, sticky='ew', padx=10, pady=5)

#pagina principala
main_frame = tk.Frame(window)
main_frame.grid(row=1, column=0, sticky='nsew', padx=10, pady=5)
main_frame.columnconfigure(0, weight=1)
main_frame.columnconfigure(1, weight=1)
main_frame.rowconfigure(0, weight=1)

#taburi
notebook = ttk.Notebook(main_frame)
notebook.grid(row=0, column=0, sticky='nsew', padx=5)

tab_raw = ttk.Frame(notebook)
tab_scaled = ttk.Frame(notebook)
tab_result = ttk.Frame(notebook)
tab_plot = ttk.Frame(notebook)

notebook.add(tab_raw, text='Date originale')
notebook.add(tab_scaled, text='Date normalizate')
notebook.add(tab_result, text='Rezultate')
notebook.add(tab_plot, text='Grafic')

tree_raw = ttk.Treeview(tab_raw)
tree_raw.pack(fill='both', expand=True)

tree_scaled = ttk.Treeview(tab_scaled)
tree_scaled.pack(fill='both', expand=True)

tree_result = ttk.Treeview(tab_result)
tree_result.pack(fill='both', expand=True)

metrics_label = tk.Label(tab_result, text="", font=("Calibri", 12))
metrics_label.pack(pady=10)

def show_dataframe_in_treeview(tree, df):
    tree.delete(*tree.get_children())
    tree["columns"] = list(df.columns)
    tree["show"] = "headings"
    for col in df.columns:
        tree.heading(col, text=col)
    for _, row in df.head(100).iterrows():
        tree.insert("", "end", values=list(row))

#menubar
menubar = tk.Menu(window)
window.config(menu=menubar)
subMenu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label="Fișiere", menu=subMenu)

def upload_csv():
    global data
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        try:
            with open(file_path, 'r') as f:
                first_line = f.readline()
                sep = '\t' if '\t' in first_line else ','
            data = pd.read_csv(file_path, sep=sep)
            messagebox.showinfo("Succes", "Fișier CSV încărcat cu succes!")
            show_dataframe_in_treeview(tree_raw, data)
        except Exception as e:
            messagebox.showerror("Eroare", f"Eroare la încărcarea CSV: {str(e)}")
    else:
        messagebox.showwarning("Atenție", "Nu s-a selectat niciun fișier!")

subMenu.add_command(label="Deschide", command=lambda: Thread(target=upload_csv).start())

#etichete titlu
title_label = ttk.Label(master=title_frame, text='Detecția stărilor de avarie', font='Calibri 24 bold')
title_label.pack(pady=(10, 2))
subtitle_label = ttk.Label(master=title_frame, text='Selectează fișierul CSV din meniul "Fișiere"', font='Calibri 14')
subtitle_label.pack(pady=(0, 10))

# selectare model
model_var = tk.StringVar()
radio_frame = tk.Frame(title_frame)
radio_frame.pack()

tk.Label(radio_frame, text="Alege modalitatea de detecție", padx=10, font=("Calibri", 12, "bold")).pack(side='left')
tk.Radiobutton(radio_frame, text="Rețea neuronală(supervizat)", variable=model_var, value="Supervizată", font=("Calibri", 12)).pack(side='left')
tk.Radiobutton(radio_frame, text="AutoEncoder(nesupervizat)", variable=model_var, value="Nesupervizată", font=("Calibri", 12)).pack(side='left')

#selectare numar epoci
epoch_frame = tk.Frame(title_frame)
epoch_frame.pack(pady=5)
tk.Label(epoch_frame, text="Număr epoci: ", font=("Calibri", 12)).pack(side='left')
epoch_entry = tk.Entry(epoch_frame, width=5)
epoch_entry.insert(0, "50")
epoch_entry.pack(side='left')

def run_model():
    global data, canvas
    try:
        if data is None:
            messagebox.showerror("Eroare", "Te rog încarcă un fișier CSV")
            return

        model_type = model_var.get()
        if not model_type:
            messagebox.showerror("Eroare", "Nu ați selectat niciun model")
            return

        try:
            epochs = int(epoch_entry.get())
        except ValueError:
            messagebox.showerror("Eroare", "Numărul de epoci nu este valid")
            return

        model_name = "supervizat" if model_type == "Supervizată" else "nesupervizat"
        model_module = importlib.import_module(model_name)

        results = model_module.run_model(data, epochs)
        accuracy, conf_matrix, loss_series, df_scaled, df_result = results[:5]
        val_loss_series = results[-1]

        # Metrici suplimentare
        if len(results) > 5:
            recall, precision, f1_score, fpr, tpr, tnr, fnr = results[5:-1]
        else:
            recall = precision = f1_score = fpr = tpr = tnr = fnr = 0

        messagebox.showinfo("Succes", f"Model {model_name} rulat cu succes!")

        show_dataframe_in_treeview(tree_scaled, df_scaled)
        show_dataframe_in_treeview(tree_result, df_result)

        # Afisare metrici
        metrics_text = f"""
        Accuracy: {accuracy:.4f}
        Recall (TPR): {recall:.4f}
        Precision: {precision:.4f}
        F1 Score: {f1_score:.4f}
        False Positive Rate (FPR): {fpr:.4f}
        True Positive Rate (TPR): {tpr:.4f}
        True Negative Rate (TNR): {tnr:.4f}
        False Negative Rate (FNR): {fnr:.4f}
        """.strip()
        metrics_label.config(text=metrics_text)

        #grafic cu matricea de confuzie și loss/val_loss

        global fig, ax
        plt.close(fig)
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        if canvas:
            canvas.get_tk_widget().destroy()
            canvas = None

        # aatrice de confuzie
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax[0])
        ax[0].set_title("Matrice de Confuzie")
        ax[0].set_xlabel("Etichetă prezisă")
        ax[0].set_ylabel("Etichetă reală")

        # Grafic loss + val_loss
        ax[1].plot(loss_series, label='Loss antrenare', color='blue')
        ax[1].plot(val_loss_series, label='Loss validare', color='orange')
        ax[1].set_title("Evoluția Loss-ului pe epoci")
        ax[1].set_xlabel("Epocă")
        ax[1].set_ylabel("Loss")
        ax[1].legend()
        ax[1].grid(True)

        canvas = FigureCanvasTkAgg(fig, master=tab_plot)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    except Exception as e:
        import traceback
        traceback.print_exc()
        messagebox.showerror("Eroare la rulare", str(e))

my_button = tk.Button(
    title_frame,
    text="Rulare model",
    command=run_model,
    font=("Calibri", 14, "bold"),
    width=15,
    height=2,
)
my_button.pack(pady=10)

def on_closing():
    if messagebox.askokcancel("Ieșire", "Sigur vrei să închizi aplicația?"):
        window.destroy()
        window.quit()

window.protocol("WM_DELETE_WINDOW", on_closing)

window.mainloop()
