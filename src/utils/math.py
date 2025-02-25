import numpy as np

def get_default_math_functions():
    """
    Get the default list of mathematical functions to apply to the numerical
    features in the dataset.
    """
    list_functions = [np.log, np.log10, np.sqrt, np.square,
                      np.arcsin, np.arccos, np.arctan,
                      np.sin, np.cos, np.tan, np.sinc]
    list_functions_names = ['log', 'log10', 'sqrt', 'square',
                            'arcsin', 'arccos', 'arctan',
                            'sin', 'cos', 'tan', 'sinc']
    return list_functions, list_functions_names
