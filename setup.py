from setuptools import setup, find_packages

setup(
    name='diagnostic-moteur-ia',
    version='0.1.0',
    description='Système de diagnostic de défaillances de moteurs électriques par IA',
    author='Votre Nom',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'pyyaml',
        'joblib',
    ],
) 