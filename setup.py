from setuptools import Command, find_packages, setup

__lib_name__ = "Harmonics"
__lib_version__ = "0.0.1"
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
    packages = ['Harmonics'],
    zip_safe = False,
    include_package_data = True,
)