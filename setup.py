from setuptools import setup, find_packages

setup(
    name='optical_elliptical_multipole',
    version='0.1.0',
    author='Maverick S. H. Oh',
    author_email='maverick.sh.oh@gmail.com',
    description='',
    packages=find_packages(),
    install_requires=[
        'jax',
        'numpy',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
    python_requires='>=3.6',
)