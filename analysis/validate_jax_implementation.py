
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
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

def validate():
    print("Running JAX validation...")
    
    DEFAULT_PARAMS = {
        'n_sersic': 4.0,
        'R_sersic': 0.2,
        'amplitude': 0.03,
        'q': 0.8,
        'theta_ell': 0.0,
        'x0': 0.0,
        'y0': 0.0,
        'background': 0.001,
        'a_m': np.array([0.01, 0.01]), # for m=3, 4
        # 'phi_m': np.array([0.0, 0.0]),     # for m=3, 4  <-- user provided this in dict
        # The user's request had 'phi_m': np.array([0.0, 0.0]),
        # But wait, profiles2D.Elliptical_Multipole_Profile_2D signature has `m`, `a_m`, `phi_m` as separate args?
        # Let's check the signature.
        # def Elliptical_Multipole_Profile_2D(X, Y, intensity_fun, q, theta_ell, m, a_m, phi_m, x0=0.0, y0=0.0, **intensity_fun_kwargs):
    }
    
    # Extract params
    n_sersic = DEFAULT_PARAMS['n_sersic']
    R_sersic = DEFAULT_PARAMS['R_sersic']
    amplitude = DEFAULT_PARAMS['amplitude']
    q = DEFAULT_PARAMS['q']
    theta_ell = DEFAULT_PARAMS['theta_ell']
    x0 = DEFAULT_PARAMS['x0']
    y0 = DEFAULT_PARAMS['y0']
    background = DEFAULT_PARAMS['background']
    a_m = DEFAULT_PARAMS['a_m']
    # phi_m from user request
    phi_m = np.array([0.0, 0.0]) 
    m = np.array([3, 4]) # Assuming m=3, 4 based on user comment "for m=3, 4"
    
    # 1. Grid generation
    # Create a grid, say 100x100
    N = 100
    x = np.linspace(-2, 2, N)
    y = np.linspace(-2, 2, N)
    X, Y = np.meshgrid(x, y)
    
    # 2. Non-JAX execution
    print("Executing Non-JAX version...")
    # sersic signature: sersic(R, amplitude=1.0, R_sersic=1.0, n_sersic=4.0)
    # We pass kwargs to Profile_2D which passes them to intensity_fun
    
    kwargs = {
        'amplitude': amplitude,
        'R_sersic': R_sersic,
        'n_sersic': n_sersic
    }
    
    # Add background manually or via function? The profile functions don't seem to add background.
    # The DEFAULT_PARAMS has 'background', but profile functions return intensity.
    # I will add background after generation if needed, or just compare intensity.
    # User asked to validate profiles, so implies intensity.
    # The profile functions do NOT take background argument in profiles2D.py or intensity_functions.py.
    # I will ignore background for the profile generation validation, or add it to both.
    
    img_nonjax = nj_p2d.Elliptical_Multipole_Profile_2D(
        X, Y, nj_int.sersic,
        q=q, theta_ell=theta_ell,
        m=m, a_m=a_m, phi_m=phi_m,
        x0=x0, y0=y0,
        **kwargs
    )
    img_nonjax += background
    
    # 3. JAX execution
    print("Executing JAX version...")
    # First run might compile
    img_jax = j_p2d.Elliptical_Multipole_Profile_2D(
        X, Y, j_int.sersic,
        q=q, theta_ell=theta_ell,
        m=m, a_m=a_m, phi_m=phi_m,
        x0=x0, y0=y0,
        **kwargs
    )
    img_jax += background
    
    # Convert JAX array to numpy for comparison
    img_jax_np = np.array(img_jax)
    
    # 4. Comparison
    diff = img_nonjax - img_jax_np
    max_diff = np.max(np.abs(diff))
    print(f"Max difference: {max_diff}")
    
    if max_diff < 1e-12: # Strict-ish check for x64
        print("Validation PASSED (diff < 1e-12)")
    else:
        print("Validation WARNING (diff >= 1e-12)")

    # 5. Plotting
    print("Plotting results...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # ax[0]: Non-JAX
    # AsinhStretchPlot(axis, data, ...)
    AsinhStretchPlot(axes[0], img_nonjax, a=0.1) # check signature
    # AsinhStretchPlot signature: (axis, data, a=0.1, vmin=None, vmax=None, return_norm=False, *args, **kwargs)
    axes[0].set_title("Non-JAX")
    
    # ax[1]: JAX
    AsinhStretchPlot(axes[1], img_jax_np, a=0.1)
    axes[1].set_title("JAX")
    
    # ax[2]: Difference
    # Using simple imshow for difference
    im3 = axes[2].imshow(diff, origin='lower', cmap='bwr')
    plt.colorbar(im3, ax=axes[2])
    axes[2].set_title(f"Difference (Max: {max_diff:.2e})")
    
    output_dir = os.path.join(os.path.dirname(__file__), 'nonjax_jax_comparison')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'validation_comparison.pdf')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == '__main__':
    validate()
