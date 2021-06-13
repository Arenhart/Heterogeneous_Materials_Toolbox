# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 17:17:30 2021

@author: Rafael Arenhart
"""

import re

import numpy as np
from PIL import Image


def load_raw(raw_file_path):
    '''
    Takes a file path for a raw image, if it has a valid file and an apropriate
    .config file, will open it, convert into a numpy array and return the array
    along the config file and the config order, used to save the output raw
    file.
    '''
    config = {}
    config_order = []
    config_path = raw_file_path[:-4] + '.config'
    with open(config_path, mode = 'r') as config_file:
        for line in config_file:
            data = [i.strip() for i in line.split('=')]
            config[data[0]] = data[1]
            config_order.append(data[0])
    data_type = 'int' + str(int(config['pixel_bytes'])*8)
    if config['signed'] == '1':
        data_type = 'u' + data_type
    img = np.fromfile(raw_file_path, dtype = data_type)
    img.resize(int(config['depth']),
               int(config['height']),
               int(config['width']))

    return img, config, config_order

def load_bmp_files(files):
    '''
    Loads and stacks .bmp files
    '''
    config = {}
    config_order = []
    img_slice = Image.open(files[0]).convert('L').transpose(Image.TRANSPOSE)
    y, x = img_slice.size
    z = len(files)
    img = np.zeros((x,y,z), dtype = 'uint8')
    img[:,:,0] = np.array(img_slice)
    i = 1
    for f in files[1:]:
        img_slice = Image.open(f).convert('L').transpose(Image.TRANSPOSE)
        img[:,:,i] = np.array(img_slice)
        i += 1

    config['width'] = x
    config_order.append('width')
    config['height'] = y
    config_order.append('height')
    config['depth'] = z
    config_order.append('depth')
    config['pixel_bytes'] = 1
    config_order.append('pixel_bytes')
    config['bigendian'] = 0
    config_order.append('bigendian')
    config['lx'] = 1.000000
    config_order.append('lx')
    config['ly'] = 1.000000
    config_order.append('ly')
    config['lz'] = 1.000000
    config_order.append('lz')
    config['unit'] = 'mm'
    config_order.append('unit')
    config['signed'] = 0
    config_order.append('signed')

    return (img, config, config_order)


def save_raw(raw_file_path, img, config, config_order):
    '''
    Saves a .raw and a .config files. File path should end in .raw
    '''

    raw_output_path = raw_file_path[:-4] + '_output.raw'
    img.tofile(raw_output_path)

    with open(raw_file_path[:-4] + '_output.config',
                                                    mode = 'w') as config_file:
        for param in config_order:
            if param == 'width':
                config_file.write('width='+str(img.shape[2])+'\n')
            elif param == 'height':
                config_file.write('height='+str(img.shape[1])+'\n')
            elif param == 'depth':
                config_file.write('depth='+str(img.shape[0])+'\n')
            elif param == 'pixel_bytes':
                p = re.compile(r'.*?([0-9]+)')
                bits = int(p.findall(str(img.dtype))[0])
                config_file.write('pixel_bytes='+str(bits//8)+'\n')
            elif param == 'signed':
                p = re.compile(r'^u', re.IGNORECASE)
                if p.search(str(img.dtype)):
                    config_file.write('signed=1\n')
                else:
                    config_file.write('signed=0\n')
            else:
                newline = param + '=' + str(config[param])
                config_file.write(newline+'\n')