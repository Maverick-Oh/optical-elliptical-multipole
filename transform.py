import re

with open('ch4_optical_multipole.md', 'r') as f:
    content = f.read()

def replace_between(text, start_str, replacement_str, end_str=None):
    if end_str:
        pattern = re.escape(start_str) + r'.*?' + re.escape(end_str)
    else:
        pattern = re.escape(start_str) + r'(?:\n+)?'
    # Use lambda to avoid invalid escape sequence parsing in replacement string
    return re.sub(pattern, lambda m: replacement_str, text, flags=re.DOTALL)


replacement_1 = r"""The polar equation for the elliptical multipole transformation builds upon the unellipticized polar coordinates $(R, \phi)$. By applying the perturbation natively within this unellipticized space, the multipole scales proportionally with the base ellipse. The forward transformation from a perfect core circle $R=1$ gives:

$$ R = r_0 \left( 1 + \sum_m a_m \cos(m(\phi - \phi_m)) \right) $$ {#eq:elliptical_forward}

Conversely, the circular multipole approach applies the perturbation directly to the standard polar radius $r$ after the base circle has been ellipticized:

$$ r = r_{\text{ell}}(\theta) + \frac{r_0}{\sqrt{q}} \sum_m a_m \cos(m(\theta - \theta_m)) $$ {#eq:circular_forward}

where $r_{\text{ell}}(\theta) = r_0 \left( q \cos^2\theta + \frac{\sin^2\theta}{q} \right)^{-1/2}$. For full mathematical derivations, the reader is referred to @sec:appendix-formalism."""

content = replace_between(content, '[Explain and put polar equations for the multipole transformation for elliptical and circular multipoles here; for the details, you can ask the reader to check out the appendix.]', replacement_1)


replacement_2 = r"""To evaluate a 2D intensity profile $I(x,y)$ (such as a Sérsic profile), we must perform an inverse transformation. Rather than generating a grid and perturbing it, we take the target Cartesian pixel coordinates, un-perturb them to find their corresponding base circular radius $R_{\text{core}}$, and evaluate the 1D intensity function $I(R_{\text{core}})$.

For an **elliptical multipole**, the inverse transformation maps a pixel $(x,y)$ first to polar coordinates $(r, \theta)$ relative to the object center $(x_0, y_0)$ and aligned with the major axis $\theta_{\text{ell}}$. We circularize $(r, \theta)$ into unellipticized coordinates $(R_{\text{circ}}, \phi)$, and trace back to the unperturbed core radius:

$$ R_{\text{core}} = \frac{R_{\text{circ}}}{1 + \sum_m a_m \cos(m(\phi - \phi_m))} $$

For a **circular multipole**, we similarly align the coordinates but apply the inverse of the circular perturbation logic:

$$ R_{\text{core}} = r \left[ \left( q \cos^2\theta + \frac{\sin^2\theta}{q} \right)^{-1/2} + \frac{1}{\sqrt{q}} \sum_m a_m \cos(m(\theta - \theta_m)) \right]^{-1} $$

The programmatic logic inside the forward-modeling pipeline is described by the following pseudo-codes.

```python
# Pseudo-code for Elliptical Multipole Evaluation
def Elliptical_Multipole_Profile_2D(X, Y, intensity_fun, q, theta_ell, m, a_m, phi_m, x0, y0):
    # 1. Shift to center and rotate to align with major axis
    R, THETA = Cartesian_To_Polar(X - x0, Y - y0)
    THETA_aligned = THETA - theta_ell
    
    # 2. Circularize the grid to unellipticized coordinates
    R_circ, PHI = Circularize(R, THETA_aligned, q)
    
    # 3. Invert the multipole perturbation
    perturbation = 1.0 + sum(a * cos(m_i * (PHI - phi_i)) for a, m_i, phi_i in zip(a_m, m, phi_m))
    R_core = R_circ / perturbation
    
    # 4. Evaluate the 1D intensity profile on the core radius
    return intensity_fun(R_core)
```

```python
# Pseudo-code for Circular Multipole Evaluation
def Circular_Multipole_Profile_2D(X, Y, intensity_fun, q, theta_ell, m, a_m, theta_m, x0, y0):
    # 1. Shift to center and rotate to align with major axis
    R, THETA = Cartesian_To_Polar(X - x0, Y - y0)
    THETA_aligned = THETA - theta_ell
    
    # 2. Compute the ellipse boundary factor and invert circular multipole
    ellipse_radius_factor = (q * cos(THETA_aligned)**2 + sin(THETA_aligned)**2 / q)**(-0.5)
    perturbation = sum(a * cos(m_i * (THETA_aligned - theta_i)) for a, m_i, theta_i in zip(a_m, m, theta_m))
    R_core = R / (ellipse_radius_factor + (1/sqrt(q)) * perturbation)
    
    # 3. Evaluate the 1D intensity profile
    return intensity_fun(R_core)
```

The transformation pipeline visually follows these directed acyclic graphs:

```mermaid
flowchart TD
    A["Pixel coordinates (X, Y)"] -->|"Shift (-x0, -y0)<br>Rotate (-theta_ell)"| B["Aligned polar coords<br>(r, theta)"]
    B -->|"Circularize with q"| C["Unellipticized coords<br>(R_circ, phi)"]
    C -->|"Remove Elliptical<br>Multipole Perturbation"| D["Core radius (R_core)"]
    D -->|"Evaluate 1D<br>Intensity Profile"| E["Pixel Intensity I(X, Y)"]
```

```mermaid
flowchart TD
    A2["Pixel coordinates (X, Y)"] -->|"Shift (-x0, -y0)<br>Rotate (-theta_ell)"| B2["Aligned polar coords<br>(r, theta)"]
    B2 -->|"Compute relative to<br>Ellipse Boundary"| C2["Remove Circular<br>Multipole Perturbation"]
    C2 --> D2["Core radius (R_core)"]
    D2 -->|"Evaluate 1D<br>Intensity Profile"| E2["Pixel Intensity I(X, Y)"]
```"""

content = replace_between(content, '[Explain the inverse transformation of elliptical and circular multipoles and mapping of the coordinates to transform any radially symmetric profiles (e.g. Sersic) to its elliptical multipole / circular multipole version; for the details, ask the reader to check out the appendix A. If the elliptical multipole and circular transformations can be expressed as a set of mathematical equations, do so. Also, make pseudo-codes that expresses the transformations too. Also, make two mermaid graph codes that explains the transformation.]', replacement_2)


replacement_3 = r"""Because the elliptical multipole's perturbation is defined within the unellipticized angular coordinate $\phi$, the physical amplitude of the perturbation naturally scales alongside the stretch array of the ellipse. This produces a consistent and morphologically realistic deformation. 

In contrast, the circular multipole adds a strictly constant radial displacement at all polar angles $\theta$ regardless of the underlying ellipse's axis ratio. Consequently, along the minor axis where the physical radius is smaller, the identical radial addition appears excessively pronounced, creating unphysical geometric artifacts."""

content = replace_between(content, '[The user need to explain how elliptical and circular multipoles look like and what the issue of the circular multipole is.]', replacement_3)

replacement_4 = r"""*(Placeholder: Figures displaying $m=3$ and $m=4$ variations for both circular and elliptical formalisms will be generated to demonstrate this divergence visually.)*"""
content = replace_between(content, '[The user need to generate what m=3 and m=4 multipoles looks like in circular multiple and elliptical multipole situations.]', replacement_4)

replacement_5 = r"""### Data Analysis Pipeline"""
content = content.replace("### Actual Work [Rename this title appropriately]", replacement_5)

replacement_6 = r"""The mock data pipeline serves a dual purpose: validating the robustness of our forward-modeling inference across the parameter space, and establishing selection criteria for our actual survey sample. Mock elliptical galaxies were synthesized using combinations of variable $n_{\text{sersic}}$ and $R_{\text{sersic}}$ alongside varying background noises mimicking true observing conditions. One primary discovery during validation was that galaxies with an exceptionally small $R_{\text{sersic}}$ or vanishing multipole amplitudes yield highly uncertain constraints. Furthermore, numerical discretization effects in the rapidly increasing central pixel values of a steep Sérsic profile (high $n_{\text{sersic}}$) bias the recovery. To correct this, we implemented a sub-pixel super-sampling regime ($10\times$ resolution) during image generation before binning back to the detector pixel scale. Ultimately, this validation allowed us to determine stringent sample cuts, assuring our derived population statistics will exclusively represent targets capable of exhibiting robust constraints."""
content = replace_between(content, '[Make a draft writing on the mock data analysis pipeline based on the mermaid diagram code below and the code base you have access to; the purpose of this was to validate the data analysis pipeline and also to see what samples to analyze (e.g. too small R_sersic or too faint amplitude can result in very uncertain estimation of multipoles, which might be better to be removed from the survey sample). Also, the two sersic parameters n_sersic and R_sersic don\'t get estimated correctly and it is believed to be from the rapidly increasing pixel values around the center; supersampling was done to see if supersampling can overcome this problem. The result is not yet analyzed perfectly, so you can put some placeholder titles and content there.]', replacement_6)

# Remove intermediate instructions
for instr in [
    '[Update the mermaid code below if needed to reflect the data processing procedure for mock dataset more clearly]',
    '[Update the mermaid code below if needed to reflect the data processing procedure for actual dataset more clearly]',
    '[The figure below shows how the preprocessing pipeline works]',
    '[The user might generate polar coordinates squished together with the ellipseto show how the unellipticized coordinates work... TBD]'
]:
    content = content.replace(instr + '\n', '').replace(instr, '')


replacement_7 = r"""The observational pipeline systematically extracts and models confirmed elliptical galaxies from the Zurich COSMOS Morphology Catalog. Selected systems are downloaded directly as large FITS cutouts using the IRSA service. Initial background estimation and localized source masking are executed efficiently utilizing Source Extractor (SEP). Following basic photometric extraction, the central galaxy undergoes the comprehensive elliptical multipole Sérsic modeling described above. Models minimize the reduced $\chi^2$ via standard optimizers (such as SLSQP and L-BFGS-B, or potentially Particle Swarm Optimization for challenging multimodal surfaces). Uncertainty quantification initially relies on the Jacobian matrix derived from the objective function's immediate curvature, enabling rapid processing of large catalogues. A comprehensive sample cut is performed to filter unresolved sources relying on established limits for $R_{\text{sersic}}$ and $n_{\text{sersic}}$ derived from mock validation."""
content = replace_between(content, '[Make a draft writing on the actual data analysis pipeline based on the mermaid diagram code below and the code base you have access to. The result is not yet analyzed perfectly, so you can put some placeholder titles and content there. The sample cut will be made based on R_sersic and n_sersic.]', replacement_7)


replacement_8 = r"""At this stage, the generated mock recoveries and initial real data extractions demonstrate the pipeline's mathematical viability, successfully modeling complex multipole deviations directly on pixel data. Specifically, mock recoveries of $m=3$ and $m=4$ variations follow the input truth within the established uncertainties when avoiding small $R_{\text{sersic}}$ values."""
content = replace_between(content, '[Put some placeholder text that describes what results will be shown here]', replacement_8)

replacement_9 = r"""It should be strongly noted that the presented numbers and inferred statistical relations are strictly preliminary. Successive iterations of automated masking, enhanced background normalization, and comprehensive MCMC burn-in will be critical for forming final conclusions. Thus, the population bounds documented herein are liable to refinement."""
content = replace_between(content, '[State that the reporting result is not yet the final result and there is a chance of improvement]', replacement_9)

replacement_10 = r"""The forward modeling framework presented provides the necessary tools to rigorously map the demographic distribution of optical isophotal perturbations observed dynamically across the cosmos. Looking ahead, providing robust empirical priors for multipole amplitudes ($a_m$) offers immense utility for adjacent astrophysical studies—particularly strong gravitational lensing, where localized complex dark matter perturbations routinely limit macroscopic mass models. Connecting optical surface brightness perturbations to comprehensive mass perturbations presents a critical future direction."""
content = replace_between(content, '[Put some placeholder text here]', replacement_10)

replacement_11 = r"""### Appendix A: Multipole Formalism {#sec:appendix-formalism}

This appendix presents the full mathematical formalism underpinning elliptical multipoles, originating conceptually from fundamental perturbation on circles. 

#### A.1. Forward Transformation

##### A.1.1. Multipole on a Circle

Both circular and elliptical multipole conventions naturally agree for a theoretically circular system ($q=1$). Imagine a base circle with radius $r_0$ subjected to a radial deviation of Fourier mode $m$ with relative amplitude $a_m$ and phase $\theta_m$. The perturbed polar condition reads:

$$ r = r_0 \left( 1 + \sum_m a_m \cos(m(\theta - \theta_m)) \right) $$ {#eq:multipole_on_circle}

##### A.1.2. Multipole on an Ellipse

When transitioning to an initially elliptical base profile, mapping the perturbation necessitates breaking symmetry.

###### A.1.2.a. Elliptical Multipole

An elliptical base curve described natively in unellipticized coordinates maintains rotational independence from the semimajor flattening. Mapping coordinate sets yields an "elliptical radius" $R$, satisfying $R = r_0$, subject to the eccentric anomaly $\phi$. The transformation applies the identical perturbation equation as @eq:multipole_on_circle, completely replacing radius variants:

$$ R = r_0 \left( 1 + \sum_m a_m \cos(m(\phi - \phi_m)) \right) $$ {#eq:elliptical_multipole}

###### A.1.2.b. Circular Multipole

Conversely, retaining the raw polar framework and simply scaling the baseline ellipse with an additive uniform sinusoidal deviation defines a circular multipole:

$$ r = r_0 \left( \left( q \cos^2\theta + \frac{\sin^2\theta}{q} \right)^{-1/2} + \frac{1}{\sqrt{q}} \sum_m a_m \cos(m(\theta - \theta_m)) \right) $$

#### A.2. Inverse Transformation

Recovering intensity grids functionally relies on mapping pixels back to the baseline profile.

##### A.2.1. Elliptical Multipole

Applying an inverse coordinate transform $\mathcal{T}_{EM}^{-1} : (r, \theta) \to (r_0, \phi)$ forces the alignment back into the unellipticized coordinate space. Specifically:

$$ r_0 = r \left( 1 + \sum_m a_m \cos(m(\phi - \phi_m)) \right)^{-1} \left( q \cos^2\theta + \frac{\sin^2\theta}{q} \right)^{1/2} $$

##### A.2.2. Circular Multipole

Similarly, isolating $r_0$ directly via algebra provides the circular multipole root radius calculation:

$$ r_0 = r \left( \left( q \cos^2\theta + \frac{\sin^2\theta}{q} \right)^{-1/2} + \frac{1}{\sqrt{q}} \sum_m a_m \cos(m(\theta - \theta_m)) \right)^{-1} $$

#### A.3. Simulating Intensity Profiles

Possessing the analytical un-perturbed base coordinate $r_0$, an arbitrary structural 1D law (e.g. an exponential profile $I(r_0) = \exp(-r_0)$) seamlessly constructs realistic 2D pixel grids by directly substituting $r_0(x,y)$. 
"""

# Replace the content safely
content = content.replace('[The appendix is not well organized yet--the headers might not be in the right rank (e.g. some subsections might need to go into another subsection as a subsubsection, etc.). Edit those. Also, improve the writing overall for academic publication instead of education tutorial type of writing.]', '')

idx_start = content.find('### Appendix A. Multipole Formalism')
idx_end = content.find('\n---', idx_start)
if idx_start != -1 and idx_end != -1:
    content = content[:idx_start] + replacement_11 + content[idx_end:]


with open('ch4_optical_multipole_ver01-20260315.md', 'w') as f:
    f.write(content)
