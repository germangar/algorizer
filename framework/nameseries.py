import numpy as np
from . import active

class series_c(np.ndarray):
    def __new__( cls, input_array, name:str, assignable:bool=True, index=-1 ):

        assert name is not None, f"The whole point of named series_c is to have a name"

        if input_array is None:
            if name in active.timeframe.registeredSeries.keys():
                raise ValueError( f"Can not create a new column. Series [{name}] already exists.")
            assignable = True
            index = active.timeframe.createColumn()
            input_array = active.timeframe.dataset[:, index]

        if input_array.shape[0] != active.timeframe.dataset.shape[0]:
            raise ValueError( f"series_c. Input array [{name}] lenght doesn't match the dataset lenght")

        obj = np.array(input_array, copy=False).view(cls)

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
        obj.timeframe = active.timeframe
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
        self.timeframe = getattr(obj, 'timeframe', None)

    def __getitem__(self, key):
        tf = getattr(self, 'timeframe', None)
        atf = active.timeframe
        if tf == atf:
            return super().__getitem__(key)
        
        # the series belongs to a different timeframe
        if isinstance( key, (float, int) ):
            if key < 0:
                key = active.timeframe.barindex + 1 + key
            timestamp = active.timeframe.timestamp - ((active.timeframe.barindex - key) * active.timeframe.timeframeMsec)
            key = self.timeframe.indexForTimestamp(timestamp)
            return super().__getitem__(key)

        raise ValueError( "Only single value access is allowed from a different timeframe.")
        
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
