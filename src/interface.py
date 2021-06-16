# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 17:17:30 2021

@author: Rafael Arenhart
"""

import os
import time

import numpy as np
import tkinter as tk
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
from PIL import Image, ImageTk

import src.io as io
import src.operations as operations

SRC_FOLDER = os.path.dirname(os.path.realpath(__file__))
OPERATIONS = {
        'OTSU': (operations.otsu_threshold, True),
        'WATE': (operations.watershed, True),
        'AOSE': (operations.segregator, True),
        'SHAP': (operations.shape_factor, False),
        'AAPS': (operations.AA_pore_scale_permeability, False),
        'SKEL': (operations.skeletonizer, True),
        'SBPS': (operations.SB_pore_scale_permeability, False),
        'LABL': (operations.labeling, False),
        'ESTL': (operations.export_stl, False),
        'RESC': (operations.rescale, False),
        'MCAV': (operations.marching_cubes_area_and_volume, False),
        'FFSO': (operations.formation_factor_solver, False),
        'BKDI': (operations.breakthrough_diameter, False),
        'FMCH': (operations.breakthrough_diameter, False)
        }

class Interface():

    def __init__(self):

        self.root = tk.Tk()
        self.root.title('Heterogeneous Materials Analyzer')
        self.operations_dictionary = {}
        self.strings = {}
        self.load_config()
        self.lang_file = 'hma_' + self.config['language'] + '.lng'

        with open(os.path.join(SRC_FOLDER, self.lang_file), mode = 'r') as file:
            for line in file:
                key, value = line.split(':')
                key = key.strip()
                value = value.strip()
                self.strings[key] = value

        for name, properties in OPERATIONS.items():
            self.operations_dictionary[self.get_string(name)] = {
                'function': properties[0],
                'preview': properties[1],
                'suffix': "_" +  name}

        self.selected_operation = tk.StringVar(self.root)
        self.selected_operation.set(self.get_string('SELECT_OPERATION'))
        self.selected_operation.trace(
                         'w', lambda w, x, y: self.update_op_description())
        self.operations_menu = tk.OptionMenu(self.root,
                           self.selected_operation, *tuple(self.operations_dictionary.keys()))
        self.operations_menu.config(width=50)
        self.operations_menu.pack(side = tk.TOP)

        self.op_description = tk.Message(self.root , width = 300)
        self.op_description.pack(side = tk.TOP)

        self.frm_main_buttons = tk.Frame(self.root)
        self.frm_main_buttons.pack(side=tk.TOP)

        self.btn_select_main = tk.Button(self.frm_main_buttons,
                                  text = self.get_string('MAIN_SELECT_BUTTON'),
                                  command = self.select_image,
                                  state = tk.DISABLED)
        self.btn_select_main.pack(side = tk.LEFT, padx = 10)
        self.btn_close_main = tk.Button(self.frm_main_buttons,
                                   text = self.get_string('MAIN_CLOSE_BUTTON'),
                                   command = self.root.destroy)
        self.btn_close_main .pack(side= tk.LEFT)

        self.lbl_extras = tk.Label(self.root, text = 'Extra functions')
        self.lbl_extras.pack(side = tk.TOP)
        self.frm_extra_buttons = tk.Frame(self.root)
        self.frm_extra_buttons.pack(side = tk.TOP)
        self.btn_convert_bmp = tk.Button(self.frm_extra_buttons,
                                         text = 'Convert BMP to RAW',
                                         command = self.convert_bmp_to_raw)
        self.btn_convert_bmp.pack(side=tk.LEFT, padx = 5)


    def load_config(self):

        self.config = {}
        with open(os.path.join(SRC_FOLDER, 'hma.cfg'), mode='r') as file:
            for line in file:
                key, value = line.split(':')
                key = key.strip()
                value = value.strip()
                self.config[key] = value


    def update_op_description(self):

        operation = self.selected_operation.get()
        suffix = self.operations_dictionary[operation]['suffix']
        description_string = self.get_string('DESCRIPTION' + suffix)
        self.op_description.config(text = description_string)
        self.btn_select_main.config(state = tk.ACTIVE)


    def close_all(self):

        self.top_preview.destroy()
        self.root.destroy()


    def select_image(self):

        self.root.withdraw()
        self.img_path = filedialog.askopenfilename()
        self.img, self.img_config, self.config_order = io.load_raw(self.img_path)
        self.top_preview = tk.Toplevel(self.root)
        self.top_preview.title('Preview')
        self.top_preview.protocol("WM_DELETE_WINDOW", self.close_all)
        self.cnv_preview = tk.Canvas(self.top_preview, width=200, height=200)
        self.cnv_preview.pack(side = tk.TOP)
        self.msg_preview = tk.Message(self.top_preview, width = 120)
        self.fill_text_preview(text_widget = self.msg_preview)
        self.msg_preview.pack(side = tk.TOP)
        self.dct_parameters = {}
        self.frm_preview_parameters = tk.Frame(self.top_preview)
        self.create_parameters_frame(self.frm_preview_parameters,
                                     self.dct_parameters)
        self.frm_preview_parameters.pack(side = tk.TOP)
        self.frm_preview_buttons = tk.Frame(self.top_preview)
        self.btn_preview_preview = tk.Button(self.frm_preview_buttons,
                                 text = self.get_string('BTN_PREVIEW_PREVIEW'),
                                 command = self.preview_preview)
        if not self.operations_dictionary[self.selected_operation.get()]['preview']:
            self.btn_preview_preview.config(state = tk.DISABLED)
        self.btn_preview_preview.pack(side = tk.LEFT, padx = 10)
        self.btn_preview_run = tk.Button(self.frm_preview_buttons,
                             text = self.get_string('BTN_PREVIEW_RUN'),
                             command = self.preview_run)
        self.btn_preview_run.pack(side = tk.LEFT, padx = 10)
        self.btn_preview_cancel = tk.Button(self.frm_preview_buttons,
                                  text = self.get_string('BTN_PREVIEW_CANCEL'),
                                  command = self.preview_cancel)
        self.btn_preview_cancel.pack(side = tk.LEFT, padx = 10)
        self.frm_preview_buttons.pack(side = tk.TOP)
        self.preview_img = None
        self.preview_vol = None
        self.create_preview_images()


    def preview_cancel(self):

        self.top_preview.withdraw()
        self.root.iconify()


    def preview_preview(self):

        op = self.selected_operation.get()
        op_suffix = self.operations_dictionary[op]['suffix']
        if op_suffix == '_OTSU':
            pre_im = operations.otsu_threshold(self.preview_vol[:,:,0])
        elif op_suffix == '_WATE':
            compactness = float(self.dct_parameters['compactness'].get())
            pre_im = operations.watershed(self.preview_vol, compactness, two_d = True)
            pre_im = pre_im[:,:,1]
        elif op_suffix == '_AOSE':
            threshold = float(self.dct_parameters['threshold'].get())
            pre_im = operations.segregator(self.preview_vol, threshold, two_d = True)
            pre_im = pre_im[:,:,1]
        elif op_suffix == '_SKEL':
            pre_im = operations.skeletonizer(self.preview_vol).sum(axis = 2)

        self.preview_img = np.array(Image.fromarray(pre_im).resize((300,300)))
        self.preview_img = (self.preview_img/np.max(self.preview_img))*254
        self.preview_img = self.preview_img.astype('int8')

        self.tk_img =  ImageTk.PhotoImage(image=
                                        Image.fromarray(self.preview_img))
        self.cnv_preview.create_image((0,0), anchor="nw", image=self.tk_img)


    def preview_run(self):

        op = self.selected_operation.get()
        op_suffix = self.operations_dictionary[op]['suffix']
        self.dct_parameters

        if op_suffix == '_OTSU':
            out_img = operations.otsu_threshold(self.img)
            io.save_raw(self.img_path[:-4]+op_suffix+'.raw',
                     out_img,
                     self.img_config,
                     self.config_order)

        elif op_suffix == '_WATE':
            try:
                compactness = float(self.dct_parameters['compactness'].get())
            except ValueError:
                messagebox.showinfo('Error', 'Entry is not a float')
                return
            out_img = operations.watershed(self.img, compactness)
            io.save_raw(self.img_path[:-4]+op_suffix+'.raw',
                         out_img,
                         self.img_config,
                         self.config_order)
        elif op_suffix == '_AOSE':
            try:
                threshold = float(self.dct_parameters['threshold'].get())
            except ValueError:
                messagebox.showinfo('Error', 'Entry is not a float')
                return
            out_img = operations.segregator(self.img, threshold)
            io.save_raw(self.img_path[:-4]+op_suffix+'.raw',
                         out_img,
                         self.img_config,
                         self.config_order)
        elif op_suffix == '_SHAP':
            factors = []
            for i in ('volume', 'surface', 'hidraulic radius',
                      'equivalent diameter', 'irregularity'):
                if self.dct_parameters[i].get() == 1:
                    factors.append(i)
            header, lines = operations.shape_factor(self.img, factors)
            with open(self.img_path[:-4]+op_suffix+'.txt', mode = 'w') as file:
                file.write(header+'\n')
                file.write(lines)

        elif op_suffix == '_SKEL':
            out_img = operations.skeletonizer(self.img)
            io.save_raw(self.img_path[:-4]+op_suffix+'.raw',
                         out_img,
                         self.img_config,
                         self.config_order)

        elif op_suffix == '_LABL':
            out_img = operations.labeling(self.img)
            io.save_raw(self.img_path[:-4]+op_suffix+'.raw',
                         out_img,
                         self.img_config,
                         self.config_order)

        elif op_suffix == '_AAPP':
            permeability = operations.AA_pore_scale_permeability(self.img)
            messagebox.showinfo('Permeability result',
                                f'Calculated permeability is {permeability.solution}')

        elif op_suffix == '_ESTL':
            save_path = self.img_path[:-4]+'.stl'
            try:
                step_size = int(self.dct_parameters['step_size'].get())
            except ValueError:
                messagebox.showinfo('Error', 'Entry is not a integer')
                return

            operations.export_stl(self.img, save_path, step_size)

        elif op_suffix == '_RESC':

            try:
                factor = float(self.dct_parameters['factor'].get())
            except ValueError:
                messagebox.showinfo('Error', 'Entry is not a float')
                return

            out_img = operations.rescale(self.img, factor)
            io.save_raw(self.img_path[:-4]+op_suffix+'.raw',
                         out_img,
                         self.img_config,
                         self.config_order)

        elif op_suffix == '_MCAV':
            start = time.perf_counter()
            areas, volumes = operations.marching_cubes_area_and_volume(self.img)
            with open(self.img_path[:-4]+op_suffix+'.txt', mode = 'w') as file:
                file.write('Index\tArea\tVolume\n')
                for i in range(1,len(areas)):
                    file.write(f'{i}\t{areas[i]}\t{volumes[i]}\n')
            print(time.perf_counter() - start)

        elif op_suffix == '_BKDI':
            start = time.perf_counter()
            step = float(self.dct_parameters['step'].get())
            diameter = operations.breakthrough_diameter(self.img, step)
            with open(self.img_path[:-4]+op_suffix+'.txt', mode = 'w') as file:
                file.write(f'{self.img_path} - Breakthrough diameter = {diameter}')
            print(time.perf_counter() - start)

        elif op_suffix == '_FMCH':
            start = time.perf_counter()
            characterizations = operations.full_morphology_characterization(self.img)
            with open(self.img_path[:-4]+op_suffix+'.txt', mode = 'w') as file:
                for key, values in characterizations.items():
                    file.write(f'{key},{str(values)[1:-1]}\n')
            print(time.perf_counter() - start)

        #elif op_suffix == '_AAPP':
        #elif op_suffix == '_SBPP':
        messagebox.showinfo('Done', 'Done')
        self.top_preview.withdraw()
        self.root.iconify()


    def create_preview_images(self):

        if self.img.shape[2] > 5:
            middle_slice = self.img.shape[2]//2
            self.preview_vol = self.img[:, :, middle_slice-2 : middle_slice+3]
        else:
            self.preview_vol = self.img.copy()
        self.preview_img = np.array(Image.fromarray(
                                   self.preview_vol[:,:,0]).resize((300,300)))
        self.preview_img = (self.preview_img/np.max(self.preview_img))*254
        self.preview_img = self.preview_img.astype('uint8')

        self.tk_img =  ImageTk.PhotoImage(image=
                                        Image.fromarray(self.preview_img))
        self.cnv_preview.create_image((0,0), anchor="nw", image=self.tk_img)


    def create_parameters_frame(self, frame, dict_parameters):

        op = self.selected_operation.get()
        op_suffix = self.operations_dictionary[op]['suffix']
        if op_suffix == '_OTSU':
            pass

        elif op_suffix == '_WATE':
            dict_parameters['compactness'] = tk.StringVar()
            self.lbl_param_threshold = tk.Label(frame, text = 'Compactness: ')
            self.ent_param_threshold = tk.Entry(frame,
                                textvariable = dict_parameters['compactness'])
            self.lbl_param_threshold.grid(row=0, column = 0)
            self.ent_param_threshold.grid(row=0, column = 1)

        elif op_suffix == '_AOSE':
            dict_parameters['threshold'] = tk.StringVar()
            self.lbl_param_threshold = tk.Label(frame, text = 'Threshold: ')
            self.ent_param_threshold = tk.Entry(frame,
                                textvariable = dict_parameters['threshold'])
            self.lbl_param_threshold.grid(row=0, column = 0)
            self.ent_param_threshold.grid(row=0, column = 1)

        elif op_suffix == '_SHAP':
            for i in ('volume', 'surface', 'hidraulic radius',
                      'equivalent diameter', 'irregularity'):
                dict_parameters[i] = tk.IntVar()
                dict_parameters[i].set(1)
                tk.Checkbutton(frame, text=i, variable=dict_parameters[i]).pack(side= tk.TOP)

        elif op_suffix == '_ESTL':
            dict_parameters['step_size'] = tk.StringVar()
            self.lbl_param_stepsize = tk.Label(frame, text = 'Step size: ')
            self.ent_param_stepsize = tk.Entry(frame,
                                  textvariable = dict_parameters['step_size'])
            self.lbl_param_stepsize.grid(row = 0, column = 0)
            self.ent_param_stepsize.grid(row = 0, column = 1)

        elif op_suffix == '_RESC':
            dict_parameters['factor'] = tk.StringVar()
            self.lbl_param_factor = tk.Label(frame, text = 'Rescaling factor: ')
            self.ent_param_factor = tk.Entry(frame,
                                    textvariable = dict_parameters['factor'])
            self.lbl_param_factor.grid(row = 0, column = 0)
            self.ent_param_factor.grid(row = 0, column = 1)

        elif op_suffix == '_BKDI':
            dict_parameters['step'] = tk.StringVar()
            dict_parameters['step'].set('0.1')
            self.lbl_param_factor = tk.Label(frame, text = 'Erosion step: ')
            self.ent_param_factor = tk.Entry(frame,
                                    textvariable = dict_parameters['step'])
            self.lbl_param_factor.grid(row = 0, column = 0)
            self.ent_param_factor.grid(row = 0, column = 1)

        elif op_suffix == '_FFSO':
            #TODO
            pass
        #elif op_suffix == '_AAPP':
        #elif op_suffix == '_AOSE':
        #elif op_suffix == '_SKEL':
        #elif op_suffix == '_SBPP':


    def fill_text_preview(self, text_widget):

        name = self.img_path.split('/')[-1]
        dtype = str(self.img.dtype)
        if 'int' in dtype and self.img.max() <= 1 and self.img.min() >=0:
            binary = ' (binary)'
        else:
            binary = ''
        shape = str(self.img.shape)
        text_widget.config(text=f'{name}\n{dtype} {binary}\n{shape}')


    def get_string(self, str_key):

        if str_key in self.strings:
            return self.strings[str_key]
        else:
            print('Missing string: ' + str_key)
            return str_key


    def convert_bmp_to_raw(self):

        files = filedialog.askopenfilenames()
        img, config, config_order = io.load_bmp_files(files)
        out_path = ''
        for i in zip(files[0], files[-1]):
            if i[0] == i[1]:
                out_path += i[0]
            else:
                break
        out_path += '.raw'
        io.save_raw(out_path, img, config, config_order)
        messagebox.showinfo('Done converting',
                     f'Raw image with config saved as {out_path}')

def start():
    interface = Interface()
    interface.root.mainloop()
