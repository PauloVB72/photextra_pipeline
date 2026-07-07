from setuptools import setup, find_packages

setup(
    name="photextra_pipeline",
    version="0.1.0",
    description="Multiband aperture photometry pipeline for galaxies (mergers)",
    author="Paulo",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy",
        "astropy",
        "matplotlib",
        "pyyaml",
        "sep",
        "reproject",
        "astroquery",
        "photutils",
        "rich",
    ],
    entry_points={
        "console_scripts": [
            "photextra=photextra_pipeline.cli:main",
        ],
    },
    include_package_data=True,
)
