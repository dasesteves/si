from setuptools import setup, find_packages
import os
import sys

# Adiciona o diretório src ao PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

setup(
    name='si',
    version='0.0.1',
    python_requires='>=3.7',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    zip_safe=False,
    install_requires=['numpy', 'pandas', 'scipy'],
    author='',
    author_email='',
    description='Sistemas inteligentes',
    license='Apache License Version 2.0',
    keywords='',
)