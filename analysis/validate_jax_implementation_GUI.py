
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import datetime
import jax

# Enable x64 for precision comparison
jax.config.update("jax_enable_x64", True)

# Add package root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optical_elliptical_multipole.nonjax import profiles2D as nj_p2d
from optical_elliptical_multipole.nonjax import intensity_functions as nj_int
from optical_elliptical_multipole.jax import profiles2D as j_p2d
from optical_elliptical_multipole.jax import intensity_functions as j_int
from optical_elliptical_multipole.plotting.plot_tools import AsinhStretchPlot

class GUIValidator:
    def __init__(self, root):
        self.root = root
        self.root.title("JAX Implementation Validator")
        self.root.geometry("1600x900")

        # Layout: Left frame for controls, Right frame for plots
        self.left_frame = ttk.Frame(root, padding="10")
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        self.right_frame = ttk.Frame(root, padding="10")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Plot setup
        self.fig = Figure(figsize=(12, 5), dpi=100)
        self.axs = self.fig.subplots(1, 3)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.right_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Save Button
        self.btn_save = ttk.Button(self.left_frame, text="Save PDF", command=self.save_pdf)
        self.btn_save.pack(side=tk.BOTTOM, pady=20, fill=tk.X)

        # Parameters
        self.params = {
            'n_sersic': tk.DoubleVar(value=4.0),
            'R_sersic': tk.DoubleVar(value=0.2),
            'amplitude': tk.DoubleVar(value=0.03),
            'q': tk.DoubleVar(value=0.8),
            'theta_ell': tk.DoubleVar(value=0.0),
            'x0': tk.DoubleVar(value=0.0),
            'y0': tk.DoubleVar(value=0.0),
            'background': tk.DoubleVar(value=0.001),
            'a_m3': tk.DoubleVar(value=0.01),
            'a_m4': tk.DoubleVar(value=0.01),
            'phi_m3': tk.DoubleVar(value=0.0),
            'phi_m4': tk.DoubleVar(value=0.0),
        }

        # Slider configurations (label, min, max, resolution)
        slider_configs = [
            ('n_sersic', 0.5, 10.0, 0.1),
            ('R_sersic', 0.01, 2.0, 0.01),
            ('amplitude', 0.001, 1.0, 0.001),
            ('q', 0.1, 1.0, 0.01),
            ('theta_ell', -np.pi, np.pi, 0.01),
            ('x0', -1.0, 1.0, 0.01),
            ('y0', -1.0, 1.0, 0.01),
            ('background', 0.0, 0.1, 0.0001),
            ('a_m3', -0.1, 0.1, 0.001),
            ('phi_m3', -np.pi, np.pi, 0.01),
            ('a_m4', -0.1, 0.1, 0.001),
            ('phi_m4', -np.pi, np.pi, 0.01),
        ]

        # Create Sliders
        for label, min_val, max_val, res in slider_configs:
            self.create_slider(label, min_val, max_val, res)

        # Initial plot
        self.update_plot()

    def create_slider(self, label, min_val, max_val, res):
        frame = ttk.Frame(self.left_frame)
        frame.pack(fill=tk.X, pady=5)
        
        lbl = ttk.Label(frame, text=label, width=15)
        lbl.pack(side=tk.LEFT)
        
        slider = tk.Scale(frame, from_=min_val, to=max_val, resolution=res, 
                          orient=tk.HORIZONTAL, variable=self.params[label],
                          command=self.on_slider_change)
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def on_slider_change(self, event):
        self.update_plot()

    def update_plot(self):
        # 1. Gather params
        p = {k: v.get() for k, v in self.params.items()}
        
        # Grid
        N = 100
        x = np.linspace(-2, 2, N)
        y = np.linspace(-2, 2, N)
        X, Y = np.meshgrid(x, y)
        
        kwargs = {
            'amplitude': p['amplitude'],
            'R_sersic': p['R_sersic'],
            'n_sersic': p['n_sersic']
        }
        
        m = np.array([3, 4])
        a_m = np.array([p['a_m3'], p['a_m4']])
        phi_m = np.array([p['phi_m3'], p['phi_m4']])
        
        # 2. Compute Non-JAX
        img_nonjax = nj_p2d.Elliptical_Multipole_Profile_2D(
            X, Y, nj_int.sersic,
            q=p['q'], theta_ell=p['theta_ell'],
            m=m, a_m=a_m, phi_m=phi_m,
            x0=p['x0'], y0=p['y0'],
            **kwargs
        )
        img_nonjax += p['background']

        # 3. Compute JAX
        img_jax = j_p2d.Elliptical_Multipole_Profile_2D(
            X, Y, j_int.sersic,
            q=p['q'], theta_ell=p['theta_ell'],
            m=m, a_m=a_m, phi_m=phi_m,
            x0=p['x0'], y0=p['y0'],
            **kwargs
        )
        img_jax += p['background']
        img_jax_np = np.array(img_jax)
        
        # 4. Diff
        diff = img_nonjax - img_jax_np
        max_diff = np.max(np.abs(diff))

        # 5. Plot
        for ax in self.axs:
            ax.clear()
            
        AsinhStretchPlot(self.axs[0], img_nonjax, a=0.1)
        self.axs[0].set_title("Non-JAX")
        
        AsinhStretchPlot(self.axs[1], img_jax_np, a=0.1)
        self.axs[1].set_title("JAX")
        
        im3 = self.axs[2].imshow(diff, origin='lower', cmap='bwr')
        self.axs[2].set_title(f"Diff (Max: {max_diff:.2e})")
        # Ensure aspect ratio is correct and remove axis ticks for cleaner look if needed
        # but maintaining standard look for now
        
        self.fig.tight_layout()
        self.canvas.draw()

    def save_pdf(self):
        output_dir = os.path.join(os.path.dirname(__file__), 'nonjax_jax_comparison')
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f'gui_snapshot_{timestamp}.pdf')
        
        self.fig.savefig(output_path)
        print(f"Saved snapshot to {output_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = GUIValidator(root)
    root.mainloop()
