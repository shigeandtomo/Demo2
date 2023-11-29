#!/usr/bin/python3
import warnings

import numpy as np

# from ..unit import (
#     EnergyConversion,
#     ForceConversion,
#     LengthConversion,
#     PressureConversion,
# )

from abc import ABC

from scipy import constants

AVOGADRO = constants.Avogadro  # Avagadro constant
ELE_CHG = constants.elementary_charge  # Elementary Charge, in C
BOHR = constants.value("atomic unit of length")  # Bohr, in m
HARTREE = constants.value("atomic unit of energy")  # Hartree, in Jole
RYDBERG = constants.Rydberg * constants.h * constants.c  # Rydberg, in Jole

# energy conversions
econvs = {
    "eV": 1.0,
    "hartree": HARTREE / ELE_CHG,
    "kJ_mol": 1 / (ELE_CHG * AVOGADRO / 1000),
    "kcal_mol": 1 / (ELE_CHG * AVOGADRO / 1000 / 4.184),
    "rydberg": RYDBERG / ELE_CHG,
    "J": 1 / ELE_CHG,
    "kJ": 1000 / ELE_CHG,
}

# length conversions
lconvs = {
    "angstrom": 1.0,
    "bohr": BOHR * 1e10,
    "nm": 10.0,
    "m": 1e10,
}


def check_unit(unit):
    if unit not in econvs.keys() and unit not in lconvs.keys():
        try:
            eunit = unit.split("/")[0]
            lunit = unit.split("/")[1]
            if eunit not in econvs.keys():
                raise RuntimeError(f"Invaild unit: {unit}")
            if lunit not in lconvs.keys():
                raise RuntimeError(f"Invalid unit: {unit}")
        except Exception:
            raise RuntimeError(f"Invalid unit: {unit}")


class Conversion(ABC):
    def __init__(self, unitA, unitB, check=True):
        """Parent class for unit conversion.

        Parameters
        ----------
        unitA : str
            unit to be converted
        unitB : str
            unit which unitA is converted to, i.e. `1 unitA = self._value unitB`
        check : bool
            whether to check unit validity

        Examples
        --------
        >>> conv = Conversion("foo", "bar", check=False)
        >>> conv.set_value("10.0")
        >>> print(conv)
        1 foo = 10.0 bar
        >>> conv.value()
        10.0
        """
        if check:
            check_unit(unitA)
            check_unit(unitB)
        self.unitA = unitA
        self.unitB = unitB
        self._value = 0.0

    def value(self):
        return self._value

    def set_value(self, value):
        self._value = value

    def __repr__(self):
        return f"1 {self.unitA} = {self._value} {self.unitB}"

    def __str__(self):
        return self.__repr__()


class EnergyConversion(Conversion):
    def __init__(self, unitA, unitB):
        """Class for energy conversion.

        Examples
        --------
        >>> conv = EnergyConversion("eV", "kcal_mol")
        >>> conv.value()
        23.06054783061903
        """
        super().__init__(unitA, unitB)
        self.set_value(econvs[unitA] / econvs[unitB])


class LengthConversion(Conversion):
    def __init__(self, unitA, unitB):
        """Class for length conversion.

        Examples
        --------
        >>> conv = LengthConversion("angstrom", "nm")
        >>> conv.value()
        0.1
        """
        super().__init__(unitA, unitB)
        self.set_value(lconvs[unitA] / lconvs[unitB])


class ForceConversion(Conversion):
    def __init__(self, unitA, unitB):
        """Class for force conversion.

        Parameters
        ----------
        unitA, unitB : str
            in format of "energy_unit/length_unit"

        Examples
        --------
        >>> conv = ForceConversion("kJ_mol/nm", "eV/angstrom")
        >>> conv.value()
        0.0010364269656262175
        """
        super().__init__(unitA, unitB)
        econv = EnergyConversion(unitA.split("/")[0], unitB.split("/")[0]).value()
        lconv = LengthConversion(unitA.split("/")[1], unitB.split("/")[1]).value()
        self.set_value(econv / lconv)


class PressureConversion(Conversion):
    def __init__(self, unitA, unitB):
        """Class for pressure conversion.

        Parameters
        ----------
        unitA, unitB : str
            in format of "energy_unit/length_unit^3", or in `["Pa", "pa", "kPa", "kpa", "bar", "kbar"]`

        Examples
        --------
        >>> conv = PressureConversion("kbar", "eV/angstrom^3")
        >>> conv.value()
        0.0006241509074460763
        """
        super().__init__(unitA, unitB, check=False)
        unitA, factorA = self._convert_unit(unitA)
        unitB, factorB = self._convert_unit(unitB)
        eunitA, lunitA = self._split_unit(unitA)
        eunitB, lunitB = self._split_unit(unitB)
        econv = EnergyConversion(eunitA, eunitB).value() * factorA / factorB
        lconv = LengthConversion(lunitA, lunitB).value()
        self.set_value(econv / lconv**3)

    def _convert_unit(self, unit):
        if unit == "Pa" or unit == "pa":
            return "J/m^3", 1
        elif unit == "kPa" or unit == "kpa":
            return "kJ/m^3", 1
        elif unit == "GPa" or unit == "Gpa":
            return "kJ/m^3", 1e6
        elif unit == "bar":
            return "J/m^3", 1e5
        elif unit == "kbar":
            return "kJ/m^3", 1e5
        else:
            return unit, 1

    def _split_unit(self, unit):
        eunit = unit.split("/")[0]
        lunit = unit.split("/")[1][:-2]
        return eunit, lunit

ry2ev = EnergyConversion("rydberg", "eV").value()
kbar2evperang3 = PressureConversion("kbar", "eV/angstrom^3").value()

length_convert = LengthConversion("bohr", "angstrom").value()
energy_convert = EnergyConversion("hartree", "eV").value()
force_convert = ForceConversion("hartree/bohr", "eV/angstrom").value()


def load_key(lines, key):
    for ii in lines:
        if key in ii:
            words = ii.split(",")
            for jj in words:
                if key in jj:
                    return jj.split("=")[1]
    return None


def load_block(lines, key, nlines):
    for idx, ii in enumerate(lines):
        if key in ii:
            break
    return lines[idx + 1 : idx + 1 + nlines]


def convert_celldm(ibrav, celldm):
    if ibrav == 1:
        return celldm[0] * np.eye(3)
    elif ibrav == 2:
        return celldm[0] * 0.5 * np.array([[-1, 0, 1], [0, 1, 1], [-1, 1, 0]])
    elif ibrav == 3:
        return celldm[0] * 0.5 * np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1]])
    elif ibrav == -3:
        return celldm[0] * 0.5 * np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
    else:
        warnings.warn(
            "unsupported ibrav "
            + str(ibrav)
            + " if no .cel file, the cell convertion may be wrong. "
        )
        return np.eye(3)
        # raise RuntimeError('unsupported ibrav ' + str(ibrav))


def load_cell_parameters(lines):
    blk = load_block(lines, "CELL_PARAMETERS", 3)
    ret = []
    for ii in blk:
        ret.append([float(jj) for jj in ii.split()[0:3]])
    return np.array(ret)


def load_atom_names(lines, ntypes):
    blk = load_block(lines, "ATOMIC_SPECIES", ntypes)
    return [ii.split()[0] for ii in blk]


def load_celldm(lines):
    celldm = np.zeros(6)
    for ii in range(6):
        key = "celldm(%d)" % (ii + 1)
        val = load_key(lines, key)
        if val is not None:
            celldm[ii] = float(val)
    return celldm


def load_atom_types(lines, natoms, atom_names):
    blk = load_block(lines, "ATOMIC_POSITIONS", natoms)
    ret = []
    for ii in blk:
        ret.append(atom_names.index(ii.split()[0]))
    return np.array(ret, dtype=int)


def load_param_file(fname):
    with open(fname) as fp:
        lines = fp.read().split("\n")
    natoms = int(load_key(lines, "nat"))
    ntypes = int(load_key(lines, "ntyp"))
    atom_names = load_atom_names(lines, ntypes)
    atom_types = load_atom_types(lines, natoms, atom_names)
    atom_numbs = []
    for ii in range(ntypes):
        atom_numbs.append(np.sum(atom_types == ii))
    ibrav = int(load_key(lines, "ibrav"))
    celldm = load_celldm(lines)
    if ibrav == 0:
        cell = load_cell_parameters(lines)
    else:
        cell = convert_celldm(ibrav, celldm)
    cell = cell * length_convert
    # print(atom_names)
    # print(atom_numbs)
    # print(atom_types)
    # print(cell)
    return atom_names, atom_numbs, atom_types, cell


def _load_pos_block(fp, natoms):
    head = fp.readline()
    if not head:
        # print('get None')
        return None, None
    else:
        ss = head.split()[0]
        blk = []
        for ii in range(natoms):
            newline = fp.readline()
            if not newline:
                return None, None
            blk.append([float(jj) for jj in newline.split()])
        return blk, ss


def load_data(fname, natoms, begin=0, step=1, convert=1.0):
    coords = []
    steps = []
    cc = 0
    with open(fname) as fp:
        while True:
            blk, ss = _load_pos_block(fp, natoms)
            if blk is None:
                break
            else:
                if cc >= begin and (cc - begin) % step == 0:
                    coords.append(blk)
                    steps.append(ss)
            cc += 1
    coords = convert * np.array(coords)
    return coords, steps


# def load_pos(fname, natoms) :
#     coords = []
#     with open(fname) as fp:
#         while True:
#             blk = _load_pos_block(fp, natoms)
#             # print(blk)
#             if blk == None :
#                 break
#             else :
#                 coords.append(blk)
#     coords= length_convert * np.array(coords)
#     return coords


def load_energy(fname, begin=0, step=1):
    data = np.loadtxt(fname)
    steps = []
    for ii in data[begin::step, 0]:
        steps.append("%d" % ii)
    with open(fname) as fp:
        while True:
            line = fp.readline()
            if not line:
                return None
            if line.split()[0][0] != "#":
                nw = len(line.split())
                break
    data = np.reshape(data, [-1, nw])
    return energy_convert * data[begin::step, 5], steps


# def load_force(fname, natoms) :
#     coords = []
#     with open(fname) as fp:
#         while True:
#             blk = _load_pos_block(fp, natoms)
#             # print(blk)
#             if blk == None :
#                 break
#             else :
#                 coords.append(blk)
#     coords= force_convert * np.array(coords)
#     return coords


def to_system_data(input_name, prefix, begin=0, step=1):
    data = {}
    data["atom_names"], data["atom_numbs"], data["atom_types"], cell = load_param_file(
        input_name
    )
    data["coords"], csteps = load_data(
        prefix + ".pos",
        np.sum(data["atom_numbs"]),
        begin=begin,
        step=step,
        convert=length_convert,
    )
    data["orig"] = np.zeros(3)
    try:
        data["cells"], tmp_steps = load_data(
            prefix + ".cel", 3, begin=begin, step=step, convert=length_convert
        )
        data["cells"] = np.transpose(data["cells"], (0, 2, 1))
        if csteps != tmp_steps:
            csteps.append(None)
            tmp_steps.append(None)
            for int_id in range(len(csteps)):
                if csteps[int_id] != tmp_steps[int_id]:
                    break
            step_id = begin + int_id * step
            raise RuntimeError(
                f"the step key between files are not consistent. "
                f"The difference locates at step: {step_id}, "
                f".pos is {csteps[int_id]}, .cel is {tmp_steps[int_id]}"
            )
    except FileNotFoundError:
        data["cells"] = np.tile(cell, (data["coords"].shape[0], 1, 1))
    return data, csteps


def to_system_label(input_name, prefix, begin=0, step=1):
    atom_names, atom_numbs, atom_types, cell = load_param_file(input_name)
    energy, esteps = load_energy(prefix + ".evp", begin=begin, step=step)
    force, fsteps = load_data(
        prefix + ".for",
        np.sum(atom_numbs),
        begin=begin,
        step=step,
        convert=force_convert,
    )
    assert esteps == fsteps, "the step key between files are not consistent "
    return energy, force, esteps


if __name__ == "__main__":
    prefix = "../../qe.traj/traj6"
    atom_names, atom_numbs, atom_types, cell = load_param_file(prefix + ".in")
    coords, _ = load_data(prefix + ".pos", np.sum(atom_numbs))
    cells, _ = load_data(prefix + ".cel", 3)
    data, csteps=to_system_data(
            prefix+".in", prefix, begin=0, step=1
    )
    energy, force, esteps=to_system_label(
        prefix+".in", prefix, begin=0, step=1
    )
    # print(atom_names)
    # print(atom_numbs)
    # print(atom_types)
    # print(cells)
    # print(coords.shape)
    # print(cells.shape)
    # print(data["coords"])
    # print(data["cells"])
    # print(data)
    # print(csteps)
    # print(energy)
    # print(force)
    # print(esteps)
