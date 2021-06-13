# -*- coding: utf-8 -*-
"""
@author: Rafael Arenhart
"""

import argparse
import importlib

parser = argparse.ArgumentParser()
parser.add_argument("Command")
parser.add_argument("Options", nargs='*')
parser.parse_args()
args = parser.parse_args()

def start_interface():
    interface = importlib.import_module('src.interface')
    interface.start()

commands = {
    'interface' : start_interface,
    }

commands[args.Command]()