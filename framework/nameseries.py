import numpy as np
from . import active

class series_c(np.ndarray):
    def __new__( cls, input_array, name:str, assignable:bool=True, index=-1 ):
        obj = np.array(input_array, copy=False).view(cls)

        # Asserts
        assert name is not None, f"The whole point of named series_c is to have a name"
        assert obj.base is not None, f"The NamedArray {name} is not a view of the original array"

        # Try to find the index of the array in the dataset columns
        if index == -1:
            dataset = active.timeframe.dataset
            for idx in range(dataset.shape[1]):
                if np.shares_memory(dataset[:, idx], input_array) or np.all(dataset[:, idx] == input_array):
                    index = idx

        if assignable and name.startswith('_'):
            raise ValueError( f"series_c [{name}] : Names starting with underscore are reserver for generated series." )

        # the series doesnot have a column, create one
        if index == -1:
            assignable = True
            index = active.timeframe.createColumn()

        # Add the name property to the instance
        obj.name = name
        obj.index = index
        obj.assignable = assignable

        # check if it's already registered in the timeframe, otherwise register it
        if name not in active.timeframe.registeredSeries.keys():
            active.timeframe.registeredSeries[name] = obj

        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.name = getattr(obj, 'name', None)
        self.index = getattr(obj, 'index', -1)
        self.assignable = True # getattr(obj, 'assignable', False)

    def __setitem__(self, key, value):
        if not self.assignable:
            raise ValueError( f"Assigning values to series_c [{self.name}] is not allowed.")
        super().__setitem__(key, value) # Call the parent class's __setitem__ method to actually set the value

    @staticmethod
    def get_column_index_from_array( candidate_col:np.ndarray ):
        dataset = active.timeframe.dataset
        # Try to find the index of the candidate_col in the dataset columns
        for idx in range(dataset.shape[1]):
            if np.shares_memory(dataset[:, idx], candidate_col) or np.all(dataset[:, idx] == candidate_col):
                return idx
        return None




# Create a NamedArray from a column of the 2D array
# named_column = NamedArray(data_2d[:, 0], name="first_column")
