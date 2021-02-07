from collections import defaultdict
from typing import Dict

import numpy as np

def get_data_dict(
    filename: str
) -> Dict[str, np.ndarray]:
    """
    Read a file of features into a dictionary.
    """
    data_dict = defaultdict(np.ndarray)
    with open(filename, 'r') as f:
        for line in f:
            if "[" in line:
                key = line.split()[0]
                mat = []
            elif "]" in line:
                line = line.split(']')[0]
                mat.append([float(x) for x in line.split()])
                data_dict[key]=np.array(mat)
            else:
                mat.append([float(x) for x in line.split()])
    return data_dict
