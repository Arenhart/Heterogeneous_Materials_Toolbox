# -*- coding: utf-8 -*-
"""
@author: Rafael Arenhart
"""

import argparse
import importlib

import src.io as io
import src.operations as operations

parser = argparse.ArgumentParser()
parser.add_argument("Command")
parser.add_argument("Options", nargs='*')
parser.parse_args()
args = parser.parse_args()

def start_interface():
    interface = importlib.import_module('src.interface')
    interface.start()
    
def bmp2raw():
    interface = importlib.import_module('src.interface')
    files = args.Options
    img, config, config_order = io.load_bmp_files(files)
    out_path = ''
    for i in zip(files[0], files[-1]):
        if i[0] == i[1]:
            out_path += i[0]
        else:
            break
    out_path += '.raw'
    io.save_raw(out_path, img, config, config_order)
    print (f'Done converting Raw image with config saved as {out_path}')

def otsu():
    img_path = args.Options[0]
    img, config, config_order = io.load_raw(img_path)
    out_img = operations.otsu_threshold(img)
    io.save_raw(
        img_path[:-4]+'_OTSU.raw',
        out_img,
        config,
        config_order
        )
    
def watershed():
    img_path = args.Options[0]
    img, config, config_order = io.load_raw(img_path)
    compactness = float(args.Options[1])
    out_img = operations.watershed(img, compactness)
    io.save_raw(
        img_path[:-4]+'_WATE.raw',
        out_img,
        config,
        config_order
        )

def segregator():
    img_path = args.Options[0]
    img, config, config_order = io.load_raw(img_path)
    threshold = float(args.Options[1])
    out_img = operations.segregator(img, threshold)
    io.save_raw(
        img_path[:-4]+'_AOSE.raw',
        out_img,
        config,
        config_order
        )

def shape_factors():
    img_path = args.Options[0]
    img, config, config_order = io.load_raw(img_path)
    factors = ('volume', 'surface', 'hidraulic radius',
              'equivalent diameter', 'irregularity')
    header, lines = operations.shape_factor(img, factors)
    with open(img_path[:-4]+'_SHAP.txt', mode = 'w') as file:
        file.write(header+'\n')
        file.write(lines)

def skeletonizer():
    img_path = args.Options[0]
    img, config, config_order = io.load_raw(img_path)
    out_img = operations.skeletonizer(img)
    io.save_raw(img_path[:-4]+'_SKEL.raw',
                 out_img,
                 config,
                 config_order)

def labeling():
    img_path = args.Options[0]
    img, config, config_order = io.load_raw(img_path)
    out_img = operations.labeling(img)
    io.save_raw(img_path[:-4]+'_LABL.raw',
                 out_img,
                 config,
                 config_order)

def arns_adler_permeability():
    img_path = args.Options[0]
    img, config, config_order = io.load_raw(img_path)
    permeability = operations.AA_pore_scale_permeability(img)
    with open(img_path[:-4]+'_AAPP.txt', mode = 'w') as file:
        file.write(
            f'Permeability result: Calculated permeability is {permeability.solution}'
            )

def export_stl():
    img_path = args.Options[0]
    img, config, config_order = io.load_raw(img_path)
    save_path = img_path[:-4]+'.stl'
    try:
        step_size = int(args.Options[1])
    except ValueError:
        print('Error, entry is not an integer')
        return
    operations.export_stl(img, save_path, step_size)

def rescale():
    img_path = args.Options[0]
    img, config, config_order = io.load_raw(img_path)
    try:
        factor = float(args.Options[1])
    except ValueError:
        print('Error, entry is not a float')
        return
    out_img = operations.rescale(img, factor)
    io.save_raw(img_path[:-4]+'_RESC.raw',
                 out_img,
                 config,
                 config_order)

def marching_cubes():
    img_path = args.Options[0]
    img, config, config_order = io.load_raw(img_path)
    areas, volumes = operations.marching_cubes_area_and_volume(img)
    with open(img_path[:-4]+'_MCAV.txt', mode = 'w') as file:
        file.write('Index\tArea\tVolume\n')
        for i in range(1,len(areas)):
            file.write(f'{i}\t{areas[i]}\t{volumes[i]}\n')

def breakthrough_diameter():
    img_path = args.Options[0]
    img, config, config_order = io.load_raw(img_path)
    step = float(args.Options[1])
    diameter = operations.breakthrough_diameter(img, step)
    with open(img_path[:-4]+'_BKDI.txt', mode = 'w') as file:
        file.write(f'{img_path} - Breakthrough diameter = {diameter}')

def morphology_characterization():
    img_path = args.Options[0]
    img, config, config_order = io.load_raw(img_path)
    characterizations = operations.full_morphology_characterization(img)
    with open(img_path[:-4]+'_FMCH.txt', mode = 'w') as file:
        for key, values in characterizations.items():
            file.write(f'{key},{str(values)[1:-1]}\n')


commands = {
    'interface' : start_interface,
    'bmp2raw' : bmp2raw,
    'otsu' : otsu,
    'watershed' : watershed,
    'segregator' : segregator,
    'shape_factors' : shape_factors,
    'skeletonizer' : skeletonizer,
    'labeling' : labeling,
    'arns_adler_permeability' : arns_adler_permeability,
    'export_stl' : export_stl,
    'rescale' : rescale,
    'marching_cubes' : marching_cubes,
    'breakthrough_diameter' : breakthrough_diameter,
    'morphology_characterization' : morphology_characterization
    }


commands[args.Command]()
