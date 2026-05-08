from setuptools import setup, find_packages


setup(

    name="pupildecompose",

    version="1.0",

    author="Sangramjit Maity",

    author_email="sangramjitm@iiitd.ac.in",

    description=(

        "A Python package for "
        "pupil feature decomposition."
    ),

    long_description=open(
        "README.md",
        encoding="utf-8"
    ).read(),

    long_description_content_type="text/markdown",

    url="https://github.com/sangramjit/pupildecompose",

    packages=find_packages(),

    include_package_data=True,

    install_requires=[

        "numpy",

        "pandas",

        "matplotlib",

        "scipy",

        "tqdm",

        "pytest"
    ],

    classifiers=[

        "Programming Language :: Python :: 3",

        "License :: OSI Approved :: MIT License",

        "Operating System :: OS Independent",

        "Intended Audience :: Science/Research",

        "Topic :: Scientific/Engineering"
    ],

    python_requires=">=3.9",
)