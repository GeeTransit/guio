try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name="guio",
    description="Guio - Curio-Tkinter Compatible Kernel",
    long_description="Guio is a library offering a curio kernel with tkinter support.",
    license="MIT",
    version="0.10",
    author="George Zhang",
    author_email="geetransit@example.com",
    url="https://github.com/GeeTransit/guio",
    packages=["guio"],
    install_requires=["curio"],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
