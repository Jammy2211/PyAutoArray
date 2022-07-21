from typing import Dict, List, Optional, Type, ValuesView

def cls_list_from(dict_values : ValuesView, cls: Type, cls_filtered: Optional[Type] = None) -> List:
    """
    Returns a list of objects in a class which are an instance of the input `cls`.

    The optional `cls_filtered` input removes classes of an input instance type.

    For example:

    - If the input is `cls=aa.pix.Pixelization`, a list containing all pixelizations in the class are returned.

    - If `cls=aa.pix.Pixelization` and `cls_filtered=aa.pix.Rectangular`, a list of all pixelizations
    excluding those which are `Rectangular` pixelizations will be returned.

    Parameters
    ----------
    dict_values
        A class dictionary of values which instances of the input `cls` are extracted from.
    cls
        The type of class that a list of instances of this class in the galaxy are returned for.
    cls_filtered
        A class type which is filtered and removed from the class list.

    Returns
    -------
        The list of objects in the galaxy that inherit from input `cls`.
    """
    if cls_filtered is not None:
        return [
            value
            for value in dict_values
            if isinstance(value, cls) and not isinstance(value, cls_filtered)
        ]
    return [value for value in dict_values if isinstance(value, cls)]