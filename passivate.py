import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Structure passivation script')

parser.add_argument('-i', '--infile', help='Input file name')
parser.add_argument('-o', '--output_type', help='Type of the output file (cfg, xyz, poscar)', default='xyz')
parser.add_argument('-t', '--input_type', help='Type of the input file (cfg)', default='cfg')

parser.add_argument('-del', '--delete', help='Delete single atoms (true or false)', default='false')

parser.add_argument('-la', '--length_a', help='Added atom bond length (angstrom)', default='1.1')

parser.add_argument('-ta', '--atom_type', help='Added atom type', default='H')

args = parser.parse_args()

is_deleted = args.delete
if is_deleted == 'true':
    is_deleted = True
else:
    is_deleted = False

infile = args.infile
outtype = args.output_type
intype = args.input_type
atom_type = args.atom_type

a_added_distance = float(args.length_a)

DEFAULT = 10

atoms_dictionary = {1: 'H', 5: 'B', 7: 'N', 9: 'F', 14: 'Si', 17: 'Cl',
                    'H': 1, 'B': 5, 'N': 7, 'F': 9, 'Si': 14, 'Cl': 17}

valency_dictionary = {'H': 1, 'B': 3, 'N': 3, 'F': 1, 'Si': 4}

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


def read_structure(filename: str) -> (list, list, list):
    structures, lattices, sizes = [], [], []

    with open(filename, 'r', encoding='utf-8') as file:

        if intype == 'cfg':
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

    if not lattices[0]:
        lattices = []
        for _ in structures:
            lattices.append([[0, 0, 0],
                             [0, 0, 0],
                             [0, 0, 0]])

    return structures, lattices, sizes


def is_in_array(component: None,
                array: list) -> bool:
    for i in array:
        if (i == component).all():
            return True
    return False


def add_atoms(structure: list,
              size: int) -> list:
    to_remove = [1]
    if is_deleted:
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


def write_structure(structures: list,
                    lattices: list) -> None:
    out = open(infile[:len(infile) - len(intype) - 1] + '_passivated.' + outtype, 'w', encoding='utf-8')

    if outtype == 'xyz':
        for i in range(len(structures)):

            xyz = f'{len(structures[i])}\n' + \
                  f'FRAME: {i} Lattice="{"{} {} {} {} {} {} {} {} {}".format(*lattices[i][0], *lattices[i][1], *lattices[i][2])}"\n'

            structures[i].sort(key=lambda x: x[1])

            for atom in structures[i]:
                xyz += f'{atoms_dictionary[atom[1]]} {"{} {} {}".format(*atom[2])}\n'

            # xyz += f'\n'

            out.write(xyz)

    elif outtype == 'cfg':
        for i in range(len(structures)):

            structures[i].sort(key=lambda x: x[1])

            cfg = f'BEGIN_CFG\n' + \
                  f' Size\n' + \
                  f'\t{len(structures[i])}\n' + \
                  f' Supercell\n' + \
                  f'\t\t{"{} {} {}".format(*lattices[i][0])}\n' + \
                  f'\t\t{"{} {} {}".format(*lattices[i][1])}\n' + \
                  f'\t\t{"{} {} {}".format(*lattices[i][2])}\n' + \
                  f' AtomData:  id type \t\tcartes_x\t\tcartes_y\t\tcartes_z\n'

            for atom in structures[i]:
                cfg += f' \t\t\t{atom[0]} \t{atom[1]} \t{"{} {} {}".format(*atom[2])}\n'

            cfg += 'END_CFG\n\n'

            out.write(cfg)

    elif outtype == 'poscar':
        for i in range(len(structures)):

            structures[i].sort(key=lambda x: x[1])

            count = [[structures[i][0][1], 1]]
            it = 0
            for k in range(1, len(structures[i])):
                if structures[i][k][1] != structures[i][k - 1][1]:
                    count.append([structures[i][k][1], 0])
                    it += 1
                count[it][1] += 1

            poscar = f'Passivated\n1\n' + \
                     f'{"{} {} {}".format(*lattices[i][0])}\n' + \
                     f'{"{} {} {}".format(*lattices[i][1])}\n' + \
                     f'{"{} {} {}".format(*lattices[i][2])}\n' + \
                     f'{("{} " * len(count)).format(*[atoms_dictionary[j[0]] for j in count])}\n' + \
                     f'{("{} " * len(count)).format(*[j[1] for j in count])}\n' + \
                     f'Cartesian\n'

            for atom in structures[i]:
                poscar += f'{"{} {} {}".format(*atom[2])}\n'

            poscar += f'\n'

            out.write(poscar)

        if len(structures) > 1:
            print('Warning: POSCAR file format does not support trajectories,\n'
                  'the output file might need postprocessing')

    out.close()


if __name__ == '__main__':
    structures, lattices, sizes = read_structure(infile)

    passivated_structures = []
    for i in range(len(sizes)):
        passivated_structures.append(add_atoms(structures[i], sizes[i]))

    write_structure(passivated_structures, lattices)
