#!/usr/bin/env python3
# coding: utf-8

from setuptools import setup

setup(
    name="PFE",
    version="0.1",
    license="MIT",
    python_requires="==3.8.16",
    zip_safe=False,
    include_package_data=True,
    packages=["Data", "Preprocessing", 'utils'],
    package_dir={
        "utils": ".",
        "Data": "./Data",
        "Preprocessing": "./Preprocessing",
    }
)
