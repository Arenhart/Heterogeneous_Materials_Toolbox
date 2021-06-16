# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 17:17:30 2021

@author: Rafael Arenhart
"""

import time
import itertools
import os
import sys

import numpy as np
import scipy as sc
from skimage import segmentation, measure, morphology, filters, transform
import stl
from numba import njit, prange

from src.Tools.conductivity_solver import ConductivitySolver
from src.Tools.rev_estudos_numba import maxi_balls
from src.Tools.jit_transport_solver import calculate_transport
from src.Tools.jit_minkowsky import get_minkowsky_functionals, get_minkowsky_functionals_parallel, minkowsky_names

PI = np.pi
SRC_FOLDER = os.path.dirname(os.path.realpath(__file__))
MC_TEMPLATES_FILE = "marching cubes templates.dat"

################
# HELPER FUNCTIONS #
################

def face_orientation(v0, v1, v2):
    '''
    Return outward perpendicular vector distance of face along the z axis
    '''
    v0 = np.array(v0),
    v1 = np.array(v1)
    v2 = np.array(v2)
    vector = np.cross(v1 - v0, v2 - v0)
    z_comp = vector[0][2]
    if z_comp > 0.1:
        return -1
    elif z_comp < -0.1:
        return 1
    else:
        return 0


def area_of_triangle(p0, p1, p2):
    '''
    As per Herons formula
    '''
    lines = list(itertools.combinations((p0, p1, p2), 2))
    distances = [(sc.spatial.distance.euclidean(i[0], i[1])) for i in lines]
    s = sum(distances)/2
    product_of_diferences = np.prod([(s-i) for i in distances])
    area = np.sqrt(s * product_of_diferences)
    return area


def mc_templates_generator(override = False):
    '''
    Generates a marching cubes template list file, if one is not available
    '''
    if MC_TEMPLATES_FILE in os.listdir(SRC_FOLDER) and not override:
        return
    summation_to_coordinate = {}
    for i in [(x, y, z) for x in range(2)
                          for y in range(2)
                          for z in range(2)]:
        summation_to_coordinate[2 ** (i[0] + 2*i[1] + 4*i[2])] = i

    templates_triangles = []
    for _ in range(256):
        templates_triangles.append( [[],[]] )

    for i in range(1,255):
        array = np.zeros((2, 2, 2))
        index = i
        for j in range(7, -1, -1):
            e = 2**j
            if index >= e:
                index -= e
                array[summation_to_coordinate[e]] = 1
        verts, faces = measure.marching_cubes_lewiner(array)[0:2]
        templates_triangles[i][0] = verts
        templates_triangles[i][1] = faces

    with open(os.path.join(SRC_FOLDER, MC_TEMPLATES_FILE), mode = 'w') as file:
        for i in range(256):
            verts, faces = templates_triangles[i]
            file.write(f'{i};')
            for v in verts:
                file.write(f'[{v[0]},{v[1]},{v[2]}]')
            file.write(';')
            for f in faces:
                file.write(f'[{f[0]},{f[1]},{f[2]}]')
            file.write('\n')


def create_mc_template_list(spacing = (1, 1, 1)):
    '''
    Return area and volume lists for the marching cubes templates
    Reads the templates file
    Input:
        Tuple with three values for x, y, and z lengths of the voxel edges
    '''
    areas = {}
    volumes = {}
    triangles = {}
    vertices_on_top = set((16, 32, 64, 128))
    with open(os.path.join(SRC_FOLDER, MC_TEMPLATES_FILE), mode = 'r') as file:
        for line in file:
            index, verts, faces = line.split(';')
            index = int(index)
            if len(verts) > 0:
                verts = verts.strip()[1:-1].split('][')
                verts = [v.split(',') for v in verts]
                verts = [[float(edge) for edge in v] for v in verts]
                faces = faces.strip()[1:-1].split('][')
                faces = [f.split(',') for f in faces]
                faces = [[int(edge) for edge in f] for f in faces]
            else:
                verts = []
                faces = []

            occupied_vertices = set()
            sub_index = index
            for i in range(7,-1,-1):
                e = 2 ** i
                if sub_index >= e:
                    occupied_vertices.add(e)
                    sub_index -= e
            total_vertices_on_top = len(occupied_vertices & vertices_on_top)
            if total_vertices_on_top == 0:
                basic_volume = 0
            elif total_vertices_on_top == 1:
                basic_volume = 1/8
            elif total_vertices_on_top == 2:
                if ((16 in occupied_vertices and 128 in occupied_vertices) or
                    (32 in occupied_vertices and 64 in occupied_vertices)):
                    basic_volume = 1/4
                else:
                    basic_volume = 1/2
            elif total_vertices_on_top == 3:
                basic_volume = 7/8
            elif total_vertices_on_top == 4:
                basic_volume = 1

            for f in faces:
                v0, v1, v2 = [verts[i] for i in f]
                v0_proj, v1_proj, v2_proj = [(i[0], i[1], 0) for i in (v0, v1, v2)]
                mean_z = sum([i[2] for i in (v0, v1, v2)])/3
                proj_area = area_of_triangle(v0_proj, v1_proj, v2_proj)
                direction = face_orientation(v0, v1, v2)
                basic_volume += mean_z * proj_area * direction

            for i in range(len(verts)):
                verts[i] = [j[0] * j[1] for j in zip(verts[i], spacing)]

            triangles[index] = (tuple(verts), tuple(faces), basic_volume)

    voxel_volume = np.prod(np.array(spacing))
    for i in triangles:
        area = 0
        verts, faces, relative_volume = triangles[i]
        for f in faces:
            triangle_area = area_of_triangle(verts[f[0]],
                                                        verts[f[1]],
                                                        verts[f[2]])
            area += triangle_area
        volume = voxel_volume * relative_volume
        areas[i] = area
        volumes[i] = volume

    return areas, volumes


def cube_generator():
    '''
    Generator yelds (x, y, z) coordinates for hollow cubes centered in (0, 0, 0)
    and edge length increasing by 2 each new cube, starting with edge
    length  equal to 3.
    '''
    x = -1
    y = -1
    z = -1
    while 1:
        out = (x, y, z)

        if abs(x) == abs(y) and abs(z) <= abs(x):
            if -abs(x) <= z and z < abs(x):
                z += 1
            elif -abs(x) <= z and z == abs(x):
                if x < 0 and y < 0:
                    z = -z
                    x += 1
                elif x > 0 and y < 0:
                    z = -z
                    x = -x
                    y += 1
                elif x < 0 and y > 0:
                    z = -z
                    x += 1
                elif x > 0 and y > 0:
                    x = -z - 1
                    y = -z - 1
                    z = -z - 1
        elif abs(x) < abs(y) and z == -abs(y):
            z += 1
        elif abs(x) < abs(y) and z == abs(y):
            z = -z
            x += 1
        elif abs(x) > abs(y) and z == -abs(x):
            z += 1
        elif abs(x) > abs(y) and z == abs(x):
            z = -z
            if x < 0:
                x += 1
            elif x > 0:
                x = -x
                y += 1
        elif z < 0 and abs(x) < abs(z) and abs(y) < abs(z):
            z = -z
        elif z > 0 and abs(x) < z and abs(y) < z:
            z = -z
            x += 1
        elif abs(x) < abs(y) and abs(z) < abs(y):
            z += 1
        elif abs(y) < abs(x) and abs(z) < abs(x):
            z += 1
        else:
            print("Error: ", x, y, z)

        yield out


def check_percolation(img):
    '''
    Returns True if binary image percolates along the z axis
    '''
    labeled = sc.ndimage.label(img)[0]
    bottom_labels = np.unique(labeled[:, :, 0])
    top_labels = np.unique(labeled[:, :, -1])
    percolating_labels = np.intersect1d(
				               bottom_labels,
                               top_labels,
                               assume_unique = True
                               )
    percolating_labels_total = (percolating_labels > 0).sum()

    return percolating_labels_total > 0


def remove_non_percolating(img):
    '''
    return image with non-percolating elements changed to 0
    '''
    labeled = sc.ndimage.label(img)[0]
    bottom_labels = np.unique(labeled[:, :, 0])
    top_labels = np.unique(labeled[:, :, -1])
    percolating_labels = np.intersect1d(
            bottom_labels,
            top_labels,
            assume_unique = True
            )
    if percolating_labels[0] == 0:
        percolating_labels = percolating_labels[1:]

    return img * np.isin(img, percolating_labels)


def wrap_sample(img, label = -1):
    '''
    Assigns "-1" to elements outside de convex hull of an image
    computed slicewise along de X axis
    '''
    print ('Wraping sample')
    x, y, z = img.shape
    outside =np.zeros((x, y, z), dtype = np.int8)
    if img.max() > 127:
        img = img // 2
    img = img.astype('int8')
    for i in range(x):
        sys.stdout.write(f"\rWraping {(100 * i / x):.2f} %")
        sys.stdout.flush()
        outside[i, :, :] = (
                np.int8(1)
                - morphology.convex_hull_image(img[i, :, :])
                )
    print()
    return img - outside


###########
# OPERATIONS #
###########

def otsu_threshold(img):

    val = filters.threshold_otsu(img)
    return (img >= val).astype('int8')


def watershed(img, compactness, two_d = False):

    if np.max(img) > 1:
        img = otsu_threshold(img)
    img[0, :, :]=0
    img[-1, :, :] = 0
    img[:, 0, :] = 0
    img[:, -1, :] = 0
    if img.shape[2] >= 3:
        img[:, :, 0] = 0
        img[:, :, -1] = 0
    else:
        x, y, z = img.shape
        temp_img = np.zeros((x, y, z+2))
        temp_img[:, :, 1:-1] = img
        img = temp_img

    tempo = time.process_time()
    print ('Start', time.process_time() - tempo)
    tempo = time.process_time()

    if two_d:
        sampling = (1, 1, 1000)
    else:
        sampling = (1, 1, 1)

    #Calcular mapa de distância
    distance_map = sc.ndimage.morphology.distance_transform_edt(
            img,
            sampling = sampling
            )

    h, w, d = img.shape
    print ('Finished distance map', time.process_time() - tempo)
    tempo = time.process_time()

    #Identificar máxmos locais
    it = ((i, j, k) for i in range(1, h-1)
                     for j in range(1, w-1)
                     for k in range(1, d-1))
    mask = np.ones((3, 3, 3))
    mask[1, 1, 1] = 0
    markers = np.zeros_like(img).astype('uint32')
    disp_it = ((i, j, k) for i in range(-1, 2)
                            for j in range(-1, 2)
                            for k in range(-1, 2))
    x, y, z = markers.shape

    for dx, dy, dz in disp_it:
        markers[1:-1, 1:-1, 1:-1] = np.maximum(
                distance_map[slice(1+dx, (-1+dx if -1+dx !=0 else None)),
                                   slice(1+dy, (-1+dy if -1+dy !=0 else None)),
                                   slice(1+dz, (-1+dz if -1+dz !=0 else None))
                                   ],
                markers[slice(1, -1),slice(1, -1),slice(1, -1)])

    markers = distance_map >= markers
    markers = markers.astype('uint32')

    print ('Finished local maxima', time.process_time()-tempo)
    tempo = time.process_time()

    #Unificar máximos agregados
    labels = sc.ndimage.label(
            markers,
            structure = sc.ndimage.generate_binary_structure(3, 3),
            output = markers
            )
    objects_box_slice = sc.ndimage.find_objects(markers)
    print(len(objects_box_slice))
    for i in range(labels):
        sl = objects_box_slice[i]
        label = i + 1
        sub_img = markers[sl]

        if sub_img.size == 1: continue

        center = [ i // 2 for i in sub_img.shape ]

        if sub_img[tuple(center)] == label:
            sub_img *= sub_img != label
            sub_img[tuple(center)] = label
            continue

        else:
            cube_it = cube_generator()
            center = np.array(center)
            while True:
                disp = np.array(next(cube_it))
                try:
                    if sub_img[tuple(center + disp)] == label:
                        sub_img *= sub_img != label
                        sub_img[tuple(center + disp)] = label
                        break
                except IndexError:
                    pass
    print ('Finished maxima aglutinator', time.process_time() - tempo)
    tempo = time.process_time()

    it = ((i, j, k) for i in range(1, h-1)
                     for j in range(1, w-1)
                     for k in range(1, d-1))
    min_radius = int(np.mean(markers>=1 * distance_map))
    for x, y, z in it:
        if markers[x, y ,z] == 0: continue
        radius = max(int(distance_map[x, y, z]), min_radius)
        sub_img = markers[x - radius:x + radius + 1,
                                   y - radius:y + radius + 1,
                                   z - radius:z + radius + 1]
        marker_distance = distance_map[x, y, z]
        if np.maximum == marker_distance:
            label = markers[x, y, z]
            lower_elements = sub_img >= label
            sub_img[:, :, :] *= lower_elements

    print ('Finished maxima mask', time.process_time()-tempo)
    tempo = time.process_time()

    #Aplicar watershed

    m = distance_map.max()
    dist_img = ((-distance_map.astype('int16') + m) ** 2).astype('uint16')
    markers = markers.astype('int32')
    out = segmentation.watershed(
            dist_img,
            markers = markers,
            mask = img,
            compactness = 1.0
            )
    print ('Finished watershed', time.process_time() - tempo)
    return out.astype('uint32')


def segregator(img, relative_threshold, two_d = False):

    print(f'Segregation using {relative_threshold} threshold.')

    if 'float' in str(img.dtype):
        img = (img / np.max(img)) * 254
        img - img.astype('int8')
    if np.max(img) > 1:
        img = otsu_threshold(img)

    h, w, d = img.shape
    tempo = time.process_time()
    print ('Start', time.process_time() - tempo)
    tempo = time.process_time()

    #Calcular mapa de distância
    if two_d:
        sampling = (1, 1, 10000)
    else:
        sampling = None
    distance_map = sc.ndimage.morphology.distance_transform_edt(
        img,
        sampling = sampling
        )
    print ('Finished distance map', time.process_time() - tempo)
    tempo = time.process_time()

    #Calcular primeiros rotulos
    label_map, max_label = sc.ndimage.label(
            img, structure = np.ones((3, 3, 3))
            )

    print ('Finished label map', time.process_time() - tempo)
    tempo = time.process_time()

    #Calcular limiar de erosao
    objects = sc.ndimage.measurements.find_objects(label_map)

    thresholds = sc.ndimage.measurements.labeled_comprehension(
            distance_map,
            labels = label_map,
            index = np.arange(1, max_label + 1),
            func = np.max,
            out_dtype = np.float,
            default = None
            )

    thresholds = np.array(thresholds) * relative_threshold
    print ('Finished local thresholds', time.process_time() - tempo)
    tempo = time.process_time()

    #Fazer fechamento seletivo
    for i in range(max_label):
        sl = distance_map[objects[i]]
        mask = label_map[objects[i]] == (i + 1)
        sl *= (((sl <= thresholds[i]) * (mask)) != 1)
        sphere = morphology.ball(thresholds[i] / 2)
        sl += sc.ndimage.morphology.binary_dilation(
                sl,
                structure=sphere,
                mask= mask
                )

    distance_map = distance_map > 0
    eroded_img = distance_map
    label_map_2, max_label_2 = sc.ndimage.label(
            eroded_img,
            structure = np.ones((3, 3, 3))
            )
    print ('Finished selective erosion', time.process_time() - tempo)
    tempo = time.process_time()

    #Recolocar elementos erodidos
    for i in range(max_label):
        if i in [int(j * max_label / 10) for j in range(10)]:
            print (int(100 * (i/max_label) + 1), r'%')

        sl = objects[i]
        th = i + 1
        _, indices = sc.ndimage.morphology.distance_transform_edt(
                (label_map_2[sl] * (label_map[sl] == th)) == 0,
                return_indices = True)

        it = ((i, j ,k) for i in range(0, sl[0].stop - sl[0].start)
                        for j in range(0, sl[1].stop - sl[1].start)
                        for k in range(0, sl[2].stop - sl[2].start))
        dilation_map = (
                (img[sl] - (label_map_2[sl] > 0))
                * (label_map[sl] == th)
                ).astype('int8')

        for x, y, z in it:
            if dilation_map[x, y, z] == 0: continue
            dx, dy, dz = indices[:, x, y, z]
            label_map_2[sl][x, y, z] = label_map_2[sl][dx, dy, dz]

    print ('Finished recovering erosion', time.process_time()-tempo)
    tempo = time.process_time()

    return label_map_2


def shape_factor(img, factors):

    '''
    'volume', 'surface', 'hidraulic radius', 'equivalent diameter', 'irregularity'
    - Volume = Número de pixeis  * (Lx*Ly*Lz); unidade = [UN^3]
    - Superfície = Resultado do marching cubes; unidade = [UN^2]
    - Raio hidráulico = Volume / Superfície; unidade = [UN]
    - Diâmetro equivalente = ((6/Pi) * Volume) ^ (1/3); unidade = [UN]
    - Irregularidade = Superfície / (Pi * Diâmetro_equivalente ^2); sem unidade
    '''
    results = ''
    header = ''
    if 'volume' in factors:
        header += 'volume\t'
    if 'surface' in factors:
        header += 'surface\t'
    if 'hidraulic radius' in factors:
        header += 'hidraulic radius\t'
    if 'equivalent diameter' in factors:
        header += 'equivalent diameter\t'
    if 'irregularity' in factors:
        header += 'irregularty\t'
    for i in factors:
        if not i in ('volume', 'surface', 'hidraulic radius',
                        'equivalent diameter', 'irregularity'):
            print(f'"{i}" factor not found')

    objects = sc.ndimage.measurements.find_objects(img)
    for i in range(0, len(objects)):
        sl = objects[i]
        label = i+1
        valid = img[sl] == label
        if min(valid.shape) <= 2:
            continue
        vol = valid.sum()
        verts, faces = measure.marching_cubes_lewiner(valid)[0:2]
        sur = measure.mesh_surface_area(verts, faces)
        eq_diam = ((6/PI) * vol) ** (0.333)
        label_eval = ''

        if 'volume' in factors:
            label_eval += str(vol)+'\t'
        if 'surface' in factors:
            label_eval += str(sur)+'\t'
        if 'hidraulic radius' in factors:
            h_r = vol / sur
            label_eval += str(h_r)+'\t'
        if 'equivalent diameter' in factors:
            e_d = ((6/PI) * vol) ** (0.333)
            label_eval += str(e_d)+'\t'
        if 'irregularity' in factors:
            irr = sur / (PI * eq_diam **2)
            label_eval += str(irr)+'\t'

        results += label_eval + '\n'

    return(header, results)


def AA_pore_scale_permeability(img):

    padded_shape = [i+2 for i in img.shape]
    padded_img = np.zeros(padded_shape, dtype = img.dtype)
    padded_img[1:-1,1:-1,1:-1] = img
    dist = sc.ndimage.morphology.distance_transform_edt(padded_img)
    dist = dist[1:-1, 1:-1, 1:-1]
    dist *= 10 * dist
    dist -= 0.5
    solver = ConductivitySolver(dist)
    solver.solve_laplacian(
            estimator_array = 'simple',
            tol = 1e-05,
            maxiter = 1e04
            )
    return solver


def formation_factor_solver(
        img,
        substitution_array = None,
        clay_labels = [],
        clay_surface_conductivity = 1.0
        ):

    '''
    calculates formation factor on a conductivity array
    '''
    if substitution_array:
        replaced_array = np.zeros(img.shape)
        for val in substitution_array:
            replaced_array += (img == val) * substitution_array[val]

        left = slice(0, -1)
        right = slice(1, None)
        full = slice(0, None)

        for sl in ( (left, full, full),
                    (right, full, full),
                    (full, right, full),
                    (full, left, full),
                    (full, full, right),
                    (full, full, left) ):

            counter_sl = tuple([ full if i == full else
                                     (right if i == left else left) for i in sl ])
            replaced_array[counter_sl] += (
                                           (replaced_array[counter_sl] > 0)
                                           * np.isin(img[sl],clay_labels)
                                           * (1 - np.isin(img[counter_sl], clay_labels))
                                           * clay_surface_conductivity
                                           )

        img = replaced_array

    solver = ConductivitySolver(img)
    solver.solve_laplacian(
            estimator_array = 'simple',
            tol = 1e-05,
            maxiter = 1e04
            )
    return solver


def skeletonizer(img):

    if 'float' in str(img.dtype):
        img = (img / np.max(img)) * 255
        img - img.astype('int8')

    if np.max(img) > 1:
        img = otsu_threshold(img)

    return morphology.skeletonize_3d(img)


def SB_pore_scale_permeability(img):
    pass


def labeling(img):
    return sc.ndimage.label(img)[0]


def export_stl(img, stl_path, step_size = 8):

    if 'float' in str(img.dtype):
        img = (img/np.max(img)) * 255
        img - img.astype('int8')

    if np.max(img) > 1:
        img = otsu_threshold(img)

    print('binary img')
    vertices, faces, _, _ =  measure.marching_cubes_lewiner(
            img,
            step_size = step_size
            )

    print('marching cubes')
    cube = stl.mesh.Mesh(
            np.zeros(
                    faces.shape[0],
                    dtype=stl.mesh.Mesh.dtype
                    )
            )

    print('mesh')
    for i, f in enumerate(faces):
        for j in range(3):
            cube.vectors[i][j] = vertices[f[j], :]
    print('cube done')
    cube.save(stl_path)


def rescale(img, factor = 0.5):

    if img.max() <= 1:
        img *= 255

    return transform.rescale(
            img,
            factor,
            multichannel = False,
            preserve_range = True,
            anti_aliasing = True
            ).astype('uint8')


def marching_cubes_area_and_volume(img, spacing = (1, 1, 1)):

    mc_templates_generator()
    X, Y, Z = img.shape
    N = img.max()
    vertex_index_array = np.array([2**i for i in range(8)])
    vertex_index_array = vertex_index_array.reshape((2, 2, 2),order = 'F')
    areas = np.zeros(N+1, dtype = 'float32')
    volumes = np.zeros(N+1, dtype = 'float32')
    template_areas, template_volumes = create_mc_template_list(spacing)

    it = ((i ,j, k) for i in range(X-1)
                     for j in range(Y-1)
                     for k in range(Z-1))

    for x, y, z in it:
        sub_array = img[x:x+2, y:y+2, z:z+2]
        labels = np.unique(sub_array)
        for label in labels:
            if label == 0: continue
            sub_interest = sub_array == label
            template_number = (sub_interest * vertex_index_array).sum()
            areas[label] += template_areas[template_number]
            volumes[label] += template_volumes[template_number]

    return areas, volumes


def breakthrough_diameter(img, step = 0.2):

    radius = 0
    dist = sc.ndimage.morphology.distance_transform_edt(img)
    while check_percolation(dist > radius):
        radius += step
    return 2 * radius


def covariogram_irregular(img):

    img = wrap_sample(img)
    x, y, z =  _covariogram_irregular(img)
    return {'x_results' : x, 'y_results' : y, 'z_results' : z}


@njit(parallel = True)
def _covariogram_irregular(img):

    print('Begin irregular covariogram')
    x, y, z = img.shape
    x_results = np.zeros(x//2, dtype = np.float64)
    y_results = np.zeros(y//2, dtype = np.float64)
    z_results = np.zeros(z//2, dtype = np.float64)

    def get_normalized_correlation(left_img, right_img):
        left_values = []
        right_values = []
        products = []
        for i in range(left_img.shape[0]):
            for j in range(left_img.shape[1]):
                for k in range(left_img.shape[2]):
                    left_val = left_img[i, j, k]
                    right_val = right_img[i, j, k]
                    if left_val == -1 or right_val == -1:
                        continue
                    left_values.append(left_val)
                    right_values.append(right_val)
                    products.append(left_val * right_val)
        if len(left_values) == 0: return None
        left_values = np.array(left_values)
        right_values = np.array(right_values)
        products = np.array(products)
        correlation = products.mean()
        product_of_expectations = left_values.mean() * right_values.mean()
        left_values.sort()
        right_values.sort()
        expectation_of_product = (left_values * right_values).mean()
        try:
            normalized_correlation = ((correlation - product_of_expectations)
                                                    / (expectation_of_product - product_of_expectations))
        except:
            normalized_correlation = 1
        return normalized_correlation

    for i in prange(1, x//2):
        left_img = img[i:, :, :]
        right_img = img[:-i, :, :]
        result = get_normalized_correlation(left_img, right_img)
        if not (result is None):
            x_results[i] = result
        else:
            break

    for i in prange(1, y//2):
        left_img = img[:, i:, :]
        right_img = img[:, :-i, :]
        result = get_normalized_correlation(left_img, right_img)
        if not (result is None):
            y_results[i] = result
        else:
            break

    for i in prange(1, z//2):
        left_img = img[:, :, i:]
        right_img = img[:, :, :-i]
        result = get_normalized_correlation(left_img, right_img)
        if not (result is None):
           z_results[i] = result
        else:
            break

    return x_results, y_results, z_results


def subsampling(img, jited_func):

    img = wrap_sample(img)
    result_length = 1

    if jited_func == _jit_pore_footprint:
        img = maxi_balls(img)

    if jited_func == _jit_permeability:
        padded_shape = [i+2 for i in img.shape]
        padded_img = np.zeros(padded_shape, dtype = img.dtype)
        padded_img[1:-1, 1:-1, 1:-1] = img
        dist = sc.ndimage.morphology.distance_transform_edt(
                padded_img
                ).astype(np.float32)
        dist = dist[1:-1, 1:-1, 1:-1]
        dist *= dist
        dist -= 0.5
        result_length = 3

    if jited_func == _jit_formation_factor:
        result_length = 3

    if jited_func == _jit_minkowsky:
        result_length = 6

    result = _subsampling(img, jited_func, result_length)
    return result


@njit(parallel = True)
def _subsampling(img, jited_func,  result_length, invalid_threshold = 0.1):

    x, y, z = img.shape
    max_radius = (min((x, y, z)) - 5) // 4
    results = np.zeros((max_radius - 1, result_length), dtype = np.float64)

    for i in prange(1, max_radius):
        minimum= i
        max_x = x - i - 1
        max_y = y - i - 1
        max_z = z - i - 1
        for _ in range(100):
            center = (np.random.randint(minimum, max_x),
                      np.random.randint(minimum, max_y),
                      np.random.randint(minimum, max_z))
            j = i + 1
            view = img[center[0] - i : center[0] + j,
                              center[1] - i : center[1] + j,
                              center[2] - i : center[2] + j]
            invalids = (view == -1).sum() / view.size
            if invalids <= invalid_threshold:
                break
        else:
            continue

        results[i, :] = jited_func(view)

    return results

@njit
def _jit_porosity(img):

    invalids = 0
    pores = 0
    x, y, z = img.shape
    for i in range(x):
        for j in range(y):
            for k in range(z):
                if img[i, j, k] == -1:
                    invalids += 1
                elif img[i, j, k] == 1:
                    pores += 1
    return pores / (img.size - invalids)

@njit
def _jit_pore_footprint(img):

    pores_n = 0
    pores_total = 0
    x, y, z = img.shape
    for i in range(x):
        for j in range(y):
            for k in range(z):
                if img[i, j, k] > 0:
                    pores_n += 1
                    pores_total += img[i, j, k]
    if pores_n == 0:
        return -1
    else:
        return pores_total / pores_n


@njit
def _jit_permeability(img):

    return calculate_transport(img)


@njit
def _jit_formation_factor(img):

    return calculate_transport(img)


@njit
def _jit_minkowsky(img):

    x, y, z = img.shape
    unwraped = img.copy()
    for i in range(x):
        for j in range(y):
            for k in range(z):
                if unwraped[i, j, k] == -1:
                    unwraped[i, j, k] == 0
    return get_minkowsky_functionals(unwraped)


def erosion_minkowsky(img):

    results = []
    eroded_img = img
    while True:
        if eroded_img.sum() == 0:
            break
        results.append(get_minkowsky_functionals_parallel(eroded_img))
        eroded_img = sc.ndimage.morphology.binary_erosion(eroded_img)

    return results


def full_morphology_characterization(img):

    img = (img>=1).astype(np.int8)
    output = {}
    #returns dictonary in form {'x_results' : x, 'y_results' : y, 'z_results' : z}
    print('Starting phase covariogram')
    start = time.perf_counter()
    result = covariogram_irregular(img)
    output['covariogram_phase_x'] = result['x_results']
    output['covariogram_phase_y'] = result['y_results']
    output['covariogram_phase_z'] = result['z_results']
    print (f'Took {time.perf_counter() - start} seconds')
    print('Starting footprint covariogram')
    start = time.perf_counter()
    result = covariogram_irregular(maxi_balls(img))
    output['covariogram_size_x'] = result['x_results']
    output['covariogram_size_y'] = result['y_results']
    output['covariogram_size_z'] = result['z_results']
    #returns list of dictionaries
    print (f'Took {time.perf_counter() - start} seconds')
    print('Starting erosion Minkowsky')
    start = time.perf_counter()
    result = erosion_minkowsky(img)
    for name in minkowsky_names:
        output[f'erosion_minkowsky_{name}'] = []
    for erosion_result in result:
        for name, value in zip(minkowsky_names, erosion_result):
            output[f'erosion_minkowsky_{name}'].append(value)
    print (f'Took {time.perf_counter() - start} seconds')
    print('Starting subsamplig Minkowsky')
    start = time.perf_counter()
    result = subsampling(img, _jit_minkowsky)
    for name in minkowsky_names:
        output[f'subsample_minkowsky_{name}'] = []
    for erosion_result in result:
        for name, value in zip(minkowsky_names, erosion_result):
            output[f'subsample_minkowsky_{name}'].append(value)
    #return a list
    print (f'Took {time.perf_counter() - start} seconds')
    print('Starting phase subsampling')
    start = time.perf_counter()
    output['subsample_phase'] = subsampling(img, _jit_porosity)
    print (f'Took {time.perf_counter() - start} seconds')
    print('Starting footprint subsample')
    start = time.perf_counter()
    output['subsample_footprint'] = subsampling(img, _jit_pore_footprint)
    #return a list of triplets
    print (f'Took {time.perf_counter() - start} seconds')
    print('Starting permeability subsampling')
    start = time.perf_counter()
    result = subsampling(img, _jit_permeability)
    output['subsampling_permeability_x'] = []
    output['subsampling_permeability_y'] = []
    output['subsampling_permeability_z'] = []
    for x, y, z in result:
        output['subsampling_permeability_x'].append(x)
        output['subsampling_permeability_y'].append(y)
        output['subsampling_permeability_z'].append(z)
    print (f'Took {time.perf_counter() - start} seconds')
    print('Starting formation factor subsampling')
    start = time.perf_counter()
    result = subsampling(img, _jit_formation_factor)
    output['subsampling_formation_factor_x'] = []
    output['subsampling_formation_factor_y'] = []
    output['subsampling_formation_factor_z'] = []
    for x, y, z in result:
        output['subsampling_formation_factor_x'].append(x)
        output['subsampling_formation_factor_y'].append(y)
        output['subsampling_formation_factor_z'].append(z)
    print (f'Took {time.perf_counter() - start} seconds')

    #expects a return of dictionary of each single variable, key i string, value is list
    return output
