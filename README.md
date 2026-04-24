# optical-elliptical-multipole
Describes elliptical multipole of galaxy light profiles in JAX (gradient-based).

Do the following for installation:
```
git clone https://github.com/Maverick-Oh/optical-elliptical-multipole.git
cd optical-elliptical-multipole
pip install .
```

You should be able to run the GUI code by:
```
python ./gui/multipole_sersic_gui.py
```

If you see errors like:
```
ImportError: dlopen(/Users/username/.local/lib/python3.10/site-packages/pyerfa-2.0.0.1-py3.10-macosx-10.9-x86_64.egg/erfa/ufunc.cpython-310-darwin.so, 0x0002): tried: '/Users/username/.local/lib/python3.10/site-packages/pyerfa-2.0.0.1-py3.10-macosx-10.9-x86_64.egg/erfa/ufunc.cpython-310-darwin.so' (mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64e' or 'arm64')), '/System/Volumes/Preboot/Cryptexes/OS/Users/username/.local/lib/python3.10/site-packages/pyerfa-2.0.0.1-py3.10-macosx-10.9-x86_64.egg/erfa/ufunc.cpython-310-darwin.so' (no such file), '/Users/username/.local/lib/python3.10/site-packages/pyerfa-2.0.0.1-py3.10-macosx-10.9-x86_64.egg/erfa/ufunc.cpython-310-darwin.so' (mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64e' or 'arm64'))
```
This means `pyerfa` (dependency of Astropy) is installed as x86_64.
You need to uninstall and reinstall pyerfa and astropy as follows:

```
pip uninstall pyerfa astropy -y
pip install --no-cache-dir --force-reinstall pyerfa astropy
```
