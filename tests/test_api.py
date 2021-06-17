# -*- coding: utf-8 -*-
"""
Created on 2021

@author: Rafael Arenhart

Run with 'python -m pytest tests\test_api.py' at HMT root
or 'python -m pytest tests\test_api.py::test_bmp2raw' for a single full test
"""

import os
import pytest
import subprocess

@pytest.mark.api
def test_bmp2raw():

    data_folder = os.path.join('tests', 'data')
    files = [os.path.join(data_folder, f'img0{i}.bmp') for i in range(5)]
    assert(subprocess.run(f'python . bmp2raw {" ".join(files)}').returncode == 0)


@pytest.mark.api
def test_otsu():

    data_folder = os.path.join('tests', 'data')
    file = "sample_gaussian.raw"
    assert(subprocess.run(f'python . otsu {os.path.join(data_folder, file)}').returncode == 0)
    