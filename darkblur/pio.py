import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union

import mrcfile
import numpy as np


def read_mrc(mrcin) -> np.array:
    return mrcfile.read(mrcin)


def extract_values_from_xml(xmlin: Union[str, Path]):
    tree = ET.parse(xmlin)
    root = tree.getroot()

    # Initialize a dictionary to store the values
    values = {'PixelSize': None, 'Defocus': None, 'Cs': None, 'Voltage': None, 'Amplitude': None}

    # Extract values
    for ctf in root.findall(".//CTF/Param"):
        name = ctf.get('Name')
        if name in values:
            values[name] = ctf.get('Value')

    return values


def modify_unselect_filter(file_path, condition):
    """
    Modifies the 'UnselectFilter' attribute in the XML file based on a given condition.

    Parameters:
    file_path (str): The path to the XML file.
    condition (bool): The condition based on which the 'UnselectFilter' value is set to True or False.
    """

    # Step 1: Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Step 2 and 3: Find the element and modify the attribute
    if 'UnselectFilter' in root.attrib:
        root.set('UnselectFilter', str(condition))

    # modified_xml_str = ET.tostring(root, encoding='unicode')
    # print(modified_xml_str)

    # Step 4: Write the modified XML back to the file
    tree.write(file_path)

