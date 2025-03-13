from setuptools import setup, find_packages

# how to release:
# UPDATE VERSION IN 3 PLACES: Ais/core/config.py, setup.py, docs/conf.py

# push to pypi:
# python setup.py sdist
# twine upload dist/*



with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='Ais-cryoET',
    version='1.0.43',
    packages=find_packages(),
    entry_points={'console_scripts': ['ais=Ais.main:main', 'ais-cryoet=Ais.main:main']},
    url='https://github.com/bionanopatterning/Ais',
    license='GPL v3',
    author='mgflast',
    author_email='m.g.f.last@lumc.nl',
    description='Segmentation of cryo-electron tomography data - https://ais-cryoet.readthedocs.org/ https://aiscryoet.org/',
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_data={'': ['*.png', '*.glsl', '*.pdf', '*.txt']},
    include_package_data=False,  # weirdly, the above filetypes _are_ included when this parameter is set to False.
    install_requires=[
        "imgui>=2.0.0",
        "tensorflow==2.8.0",
        "protobuf==3.20.0",
        "glfw>=2.5.5",
        "PyOpenGL>=3.1.6",
        "numpy>=1.23.2,<2.0",
        "mrcfile>=1.4.3",
        "Pillow>=9.2.0",
        "scipy>=1.9.1",
        "tifffile>=2022.8.12",
        "dill>=0.3.5.1",
        "pyperclip>=1.8.2",
        "scikit-image"
    ]
)
