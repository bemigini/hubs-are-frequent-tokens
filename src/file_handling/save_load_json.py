"""

For saving and loading JSON files


"""


from enum import Enum
from typing import Callable

import json
from json import JSONEncoder

import numpy as np





class CustomEncoder(JSONEncoder):
    """ A custom JSON encoder """
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, Enum):
            return o.name
        if o is None or isinstance(o, Callable):
            return ''    
        return o.__dict__


def save_as_json(obj: object, save_to: str) -> None:
    """ save object as JSON """
    json_str = json.dumps(obj, indent = 4,
                          cls = CustomEncoder)
    
    with open(save_to, 'w', encoding='utf-8') as f:
        f.write(json_str)


def load_json(load_from: str) -> str:
    """ load a JSON file """
    with open(load_from, 'r', encoding='utf-8') as f:
        json_obj = json.load(f)
        
    return json_obj
