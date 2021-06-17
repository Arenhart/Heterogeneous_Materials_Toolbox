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
    

commands = {
    'interface' : start_interface,
    'bmp2raw' : bmp2raw,
    'otsu' : otsu
    }

commands[args.Command]()
