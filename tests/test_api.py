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
    
    
@pytest.mark.api
def test_watershed():

    data_folder = os.path.join('tests', 'data')
    file = "sample_gaussian_OTSU_output.raw"
    assert(subprocess.run(f'python . watershed {os.path.join(data_folder, file)} 0.25').returncode == 0)
    
    
@pytest.mark.api
def test_segregator():

    data_folder = os.path.join('tests', 'data')
    file = "sample_gaussian_OTSU_output.raw"
    assert(subprocess.run(f'python . segregator {os.path.join(data_folder, file)} 0.25').returncode == 0)
    
    
@pytest.mark.api
def test_shape_factors():

    data_folder = os.path.join('tests', 'data')
    file = "sample_gaussian_OTSU_output_LABL_output.raw"
    assert(subprocess.run(f'python . shape_factors {os.path.join(data_folder, file)}').returncode == 0)
    
@pytest.mark.api
def test_skeletonizer():
    data_folder = os.path.join('tests', 'data')
    file = "sample_gaussian_OTSU_output.raw"
    assert(subprocess.run(f'python . skeletonizer {os.path.join(data_folder, file)}').returncode == 0)

    
@pytest.mark.api
def test_labeling():
    data_folder = os.path.join('tests', 'data')
    file = "sample_gaussian_OTSU_output.raw"
    assert(subprocess.run(f'python . labeling {os.path.join(data_folder, file)}').returncode == 0)

    
@pytest.mark.api
def test_arns_adler_permeability():
    data_folder = os.path.join('tests', 'data')
    file = "sample_gaussian_OTSU_output.raw"
    assert(subprocess.run(f'python . arns_adler_permeability {os.path.join(data_folder, file)}').returncode == 0)

    
@pytest.mark.api
def test_export_stl():
    data_folder = os.path.join('tests', 'data')
    file = "sample_gaussian_OTSU_output.raw"
    STEP_SIZE = 2
    assert(subprocess.run(f'python . export_stl {os.path.join(data_folder, file)} {STEP_SIZE}').returncode == 0)

    
@pytest.mark.api
def test_rescale():
    data_folder = os.path.join('tests', 'data')
    file = "sample_gaussian.raw"
    SCALE_FACTOR = 0.5
    assert(subprocess.run(f'python . rescale {os.path.join(data_folder, file)} {SCALE_FACTOR}').returncode == 0)

    
@pytest.mark.api
def test_marching_cubes():
    data_folder = os.path.join('tests', 'data')
    file = "sample_gaussian_OTSU_output.raw"
    assert(subprocess.run(f'python . marching_cubes {os.path.join(data_folder, file)}').returncode == 0)

    
@pytest.mark.api
def test_breakthrough_diameter():
    data_folder = os.path.join('tests', 'data')
    file = "sample_gaussian_OTSU_output.raw"
    STEP_SIZE = 2
    assert(subprocess.run(f'python . breakthrough_diameter {os.path.join(data_folder, file)} {STEP_SIZE}').returncode == 0)

    
@pytest.mark.api
def test_morphology_characterization():
    data_folder = os.path.join('tests', 'data')
    file = "sample_gaussian_OTSU_output.raw"
    assert(subprocess.run(f'python . morphology_characterization {os.path.join(data_folder, file)}').returncode == 0)

    
    