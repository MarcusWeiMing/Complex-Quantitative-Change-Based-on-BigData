
from setuptools import setup, find_packages

setup(
    name="Complex-Quantitative-Change-Based-on-Big-Data",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'scikit-learn',
        'tensorflow',
        'jupyter',
        'xgboost',
        'lightgbm',
        'catboost'
    ],
)
