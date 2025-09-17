from setuptools import Command, find_packages, setup

__lib_name__ = "harmonics-bio"
__lib_version__ = "0.0.4"
__description__ = "Hierarchical distribution matching enables comprehensive characterization of common and condition-specific cell niches in spatial omics data"
__url__ = "https://github.com/YangLabHKUST/Harmonics"
__author__ = "Yuyao Liu"
__author_email__ = "yliuow@connect.ust.hk"

setup(
    name = __lib_name__,
    version = __lib_version__,
    description = __description__,
    url = __url__,
    author = __author__,
    author_email = __author_email__,
    zip_safe = False,
    include_package_data = True,
    # long_description=open("README.md").read(),
    # long_description_content_type="text/markdown", 
    license="MIT",
    license_files=("LICENSE",),
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "anndata",
        "scipy",
        "matplotlib",
        "scikit-learn",
        "statsmodels",
        "numba",
        "tqdm",
    ],
)