# Heterogeneous Materials Toolbox

## Introduction

This is a collection of image processing and physical simulation tools targeted at porous media analysis.
Functions may be ran either with a simple GUI, or directly with CLI.

## Instalation

The toolbox requires Python 3.8 to run and may be obtained with git.

1 - Clone the repository locally with git.
`git clone https://github.com/Arenhart/Heterogeneous_Materials_Toolbox.git `

2 - Install the required libraries.
`pip install -r requirements.txt`

3 - The GUI can be started by running the module with the `interface` parameter
`python . interface`

## Commands

- bmp2raw: Stacks a series of bmp images and saves as a 3D raw file
`python . bmp2raw [BMP_IMAGE_1] [BMP_IMAGE_2] ...`

- otsu: Performs image binarization with Otsu threshold
`python . otsu [RAW_IMAGE]`
    
- watershed: Performs watershed segmentation on a binary image
`python . watershed [RAW_IMAGE] [COMPACTNESS {FLOAT}]`
    
- segregator: Perform image segmentation based in distance map thresholds
`python . segregator [RAW_IMAGE] [THRESHOLD {FLOAT}]`
    
- shape_factors: Calculate shape factors of particles in a labeled image
`python . shape_factors [RAW_IMAGE]`
    
- skeletonizer: Skeletonizes a binary image
`python . skeletonizer [RAW_IMAGE]`

- labeling: Labels continuos regions in a binary image with unique values
`python . labeling [RAW_IMAGE]`

- arns_adler_permeability: Aproximates single-phase flow with a linear algorithm
`python . arns_adler_permeability [RAW_IMAGE]`

- export_stl: Exports a binary image as an .stl file
`python . export_stl [RAW_IMAGE] [STEP_SIZE {INT}]`

- rescale: Upscales or downscales an image by a determined factor
`python . rescale [RAW_IMAGE] [SCALE_FACTOR {FLOAT}]`

- marching_cubes: Calculates volume and surface area of objects in a labeled image
`python . marching_cubes [RAW_IMAGE]`
    
- breakthrough_diameter: Calculates the flow breakthrough diameter
`python . breakthrough_diameter [RAW_IMAGE] [STEP_SIZE {INT}]`

- morphology_characterization: Performs a series of morphological characterizations on a binary image
`python . morphology_characterization [RAW_IMAGE]`