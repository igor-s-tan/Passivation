import _io
import numpy as np
import argparse
import sys
import typing

parser = argparse.ArgumentParser(description='Structure passivation script')

parser.add_argument('infile', help='Input file name')
parser.add_argument('-o', '--output_type', help='Type of the output file (cfg, xyz, poscar)', default='xyz')
parser.add_argument('-i', '--input_type', help='Type of the input file (cfg)', default='cfg')

parser.add_argument('-d', '--delete', help='Delete single atoms', action='store_true')
parser.set_defaults(delete=False)

parser.add_argument('-l', '--length_a', help='Added atom bond length (angstrom)', type=float, default='1.1')

parser.add_argument('-a', '--atom_type', help='Added atom type', default='H')

args = parser.parse_args()

is_deleted = args.delete

infile = args.infile
outtype = args.output_type
intype = args.input_type
atom_type = args.atom_type

a_added_distance = args.length_a

DEFAULT = 10

atoms_dictionary = {1: 'H', 'H': 1,
                    5: 'B', 'B': 5,
                    7: 'N', 'N': 7,
                    9: 'F', 'F': 9,
                    14: 'Si', 'Si': 14,
                    17: 'Cl', 'Cl': 17}

valency_dictionary = {'H': 1,
                      'B': 3,
                      'N': 3,
                      'F': 1,
                      'Si': 4}

cutoff_dictionary = {(1, 1): DEFAULT,
                     (1, 5): DEFAULT,
                     (1, 7): DEFAULT,
                     (1, 14): DEFAULT,
                     (5, 5): 1.74,
                     (5, 7): 2.0,
                     (5, 14): 2.1,
                     (7, 7): 2.0,
                     (7, 14): 1.98,
                     (14, 14): 2.57}


def print_error(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    
    
def read_cfg(structures: list,
             lattices: list,
             sizes: list,
             file: _io.TextIOWrapper) -> typing.Tuple[list, list, list]:
        count = -1
        for line in file:
            if 'BEGIN_CFG' in line:
                count += 1
                structures.append([])
                lattices.append([])
                continue
            string = line.split()
            if len(string) == 3:
                lattices[count].append([float(i) for i in string])
            if len(string) == 5:
                structures[count].append(
                    [int(string[0]), int(string[1]), np.array([float(string[i]) for i in range(2, 5)])])
            if len(string) > 7:
                sizes.append(int(string[2]))


def write_cfg(structures: list,
              lattices: list,
              out: _io.TextIOWrapper) -> None:
    for i in range(len(structures)):

        structures[i].sort(key=lambda x: x[0])

        cfg = f'BEGIN_CFG\n' + \
              f' Size\n' + \
              f'\t{len(structures[i])}\n' + \
              f' Supercell\n' + \
              f'\t\t{("{:10.5f} " * 3).format(*lattices[i][0])}\n' + \
              f'\t\t{("{:10.5f} " * 3).format(*lattices[i][1])}\n' + \
              f'\t\t{("{:10.5f} " * 3).format(*lattices[i][2])}\n' + \
              f' AtomData:  id type \t\tcartes_x\t\tcartes_y\t\tcartes_z\n'

        for atom in structures[i]:
            cfg += f' \t\t\t{atom[0]} \t{atom[1]} \t{("{:10.5f} " * 3).format(*atom[2])}\n'

        cfg += 'END_CFG\n\n'

        out.write(cfg)


def write_xyz(structures: list,
              lattices: list,
              out: _io.TextIOWrapper) -> None:
    for i in range(len(structures)):

        xyz = f'{len(structures[i])}\n' + \
              f'FRAME: {i} Lattice="{("{:10.5f} " * 9).format(*lattices[i][0], *lattices[i][1], *lattices[i][2])}"\n'

        structures[i].sort(key=lambda x: x[1])

        for atom in structures[i]:
            xyz += f'{atoms_dictionary[atom[1]].ljust(2, " ")} {("{:10.5f} " * 3).format(*atom[2])}\n'

        out.write(xyz)


def write_poscar(structures: list,
                 lattices: list,
                 out: _io.TextIOWrapper) -> None:
    for i in range(len(structures)):

        null_vector = [0.0, 0.0, 0.0]

        if null_vector in lattices[i]:
            print_error(f'Warning: POSCAR file format requires non-null lattice vectors,\n'
                        f'         lattice of the # {i} frame is set to cubic with A=20')
            lattices[i] = [20.0, 0.0, 0.0], \
                          [0.0, 20.0, 0.0], \
                          [0.0, 0.0, 20.0]

        structures[i].sort(key=lambda x: x[1])

        count = [[structures[i][0][1], 1]]
        it = 0
        for k in range(1, len(structures[i])):
            if structures[i][k][1] != structures[i][k - 1][1]:
                count.append([structures[i][k][1], 0])
                it += 1
            count[it][1] += 1

        poscar = f'Passivated\n1\n' + \
                 f'{("{:10.5f} " * 3).format(*lattices[i][0])}\n' + \
                 f'{("{:10.5f} " * 3).format(*lattices[i][1])}\n' + \
                 f'{("{:10.5f} " * 3).format(*lattices[i][2])}\n' + \
                 f'{("{} " * len(count)).format(*[atoms_dictionary[j[0]] for j in count])}\n' + \
                 f'{("{} " * len(count)).format(*[j[1] for j in count])}\n' + \
                 f'Cartesian\n'

        for atom in structures[i]:
            poscar += f'{("{:10.5f} " * 3).format(*atom[2])}\n'

        poscar += f'\n'

        out.write(poscar)

    if len(structures) > 1:
        print_error(f'Warning: POSCAR file format does not support trajectories,\n'
                    f'         the output file might need postprocessing\n')


input_formats = {'cfg': read_cfg}

output_formats = {'cfg': write_cfg,
                  'xyz': write_xyz,
                  'poscar': write_poscar}

def get_linear_place(r_atom: np.ndarray,
                     r_c: np.ndarray,
                     distance: float) -> np.ndarray:
    d = r_atom - r_c
    b = np.cross(np.cross(-r_atom, d), d) / np.dot(d, d)
    t1 = (np.dot(r_atom - b, d) + np.sqrt((np.dot(r_atom - b, d)) ** 2 - np.linalg.norm(d) ** 2 * (
            np.linalg.norm(r_atom) ** 2 + np.linalg.norm(b) ** 2 - 2 * np.dot(r_atom, b) - distance ** 2))) / (
                 np.linalg.norm(d) ** 2)
    t2 = (np.dot(r_atom - b, d) - np.sqrt((np.dot(r_atom - b, d)) ** 2 - np.linalg.norm(d) ** 2 * (
            np.linalg.norm(r_atom) ** 2 + np.linalg.norm(b) ** 2 - 2 * np.dot(r_atom, b) - distance ** 2))) / (
                 np.linalg.norm(d) ** 2)
    r1 = b + t1 * d
    r2 = b + t2 * d
    if np.linalg.norm(r1 - r_c) > np.linalg.norm(r2 - r_c):
        return r2
    else:
        return r1


def get_neighborhood(structure: list,
                     current_atom: list) -> list:
    neighborhood = []
    for atom in structure:
        if 0 < np.linalg.norm(current_atom[2] - atom[2]) < cutoff_dictionary[tuple(sorted([current_atom[1], atom[1]]))]:
            neighborhood.append(atom[2])
    return neighborhood


def read_structure(filename: str) -> typing.Tuple[list, list, list]:
    structures, lattices, sizes = [], [], []

    with open(filename, 'r', encoding='utf-8') as file:
        input_formats[intype](structures, lattices, sizes, file)

    if not lattices[0]:
        lattices = []
        for _ in structures:
            lattices.append([[0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0]])

    return structures, lattices, sizes


def write_structure(structures: list,
                    lattices: list) -> None:
    with open(infile[:len(infile) - len(intype) - 1] + '_passivated.' + outtype, 'w', encoding='utf-8') as file:
        output_formats[outtype](structures, lattices, file)


def is_in_array(component: None,
                array: list) -> bool:
    for i in array:
        if (i == component).all():
            return True
    return False


def remove(structure: list,
           size: int) -> typing.Tuple[list, int]:
    to_remove = [1]
    while len(to_remove) != 0:
        to_remove = []
        for i in range(size):
            new_neighborhood = get_neighborhood(structure[:size], structure[i])
            if len(new_neighborhood) <= 1 and structure[i][1] != atoms_dictionary[atom_type]:
                to_remove.append(i)

        to_remove = sorted(to_remove, reverse=True)

        for i in to_remove:
            structure.append([i + 1, structure[i][1], structure[i][2]])
            structure.pop(i)
            print(f'Atom # {i + 1} was deleted')
            size -= 1

        for i in range(len(structure)):
            structure[i][0] = i + 1

    return structure, size


def add_atoms(structure: list,
              size: int) -> list:
    if is_deleted:
        structure, size = remove(structure, size)

    atomic_places = []
    old_neighborhoods = []

    for atom in structure:
        old_neighborhoods.append(get_neighborhood(structure, atom))

    structure = structure[:size]

    for i in range(size):
        new_neighborhood = get_neighborhood(structure, structure[i])
        if abs(len(new_neighborhood) - len(old_neighborhoods[i])) > 0:
            temp = []
            for k in old_neighborhoods[i]:
                if is_in_array(k, new_neighborhood):
                    temp.append([k, np.linalg.norm(k[2] - structure[i][2]), 0])
                else:
                    temp.append([k, np.linalg.norm(k[2] - structure[i][2]), 1])
            temp.sort(key=lambda x: x[1])

            temp = temp[:valency_dictionary[atoms_dictionary[structure[i][1]]]]
            for j in temp:
                if j[2] == 1:
                    atomic_places.append(get_linear_place(structure[i][2], j[0], a_added_distance))

    for i in range(len(atomic_places)):
        structure.append([size + 1 + i, atoms_dictionary[atom_type], atomic_places[i]])

    return structure


if __name__ == '__main__':
    structures, lattices, sizes = read_structure(infile)

    passivated_structures = []
    for i in range(len(sizes)):
        passivated_structures.append(add_atoms(structures[i], sizes[i]))

    write_structure(passivated_structures, lattices)
