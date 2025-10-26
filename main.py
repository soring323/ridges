class Cell:
    def __init__(self):
        self._dependents = set()
        self._value = None
        self._callbacks = set()
        
    @property
    def value(self):
        return self._value
        
    def add_callback(self, callback):
        self._callbacks.add(callback)
        
    def remove_callback(self, callback):
        self._callbacks.discard(callback)
        
    def _notify_dependents(self):
        for dependent in self._dependents:
            dependent._invalidate()
            
    def _invalidate(self):
        """Mark cache as invalid and notify dependents"""
        old_value = self._value
        self._value = None
        if old_value is not None:
            self._notify_dependents()
            
    def _notify_callbacks(self):
        for callback in self._callbacks:
            callback(self._value)

class InputCell(Cell):
    def __init__(self, initial_value):
        super().__init__()
        self._value = initial_value
        
    @property
    def value(self):
        return self._value
        
    @value.setter
    def value(self, new_value):
        if self._value != new_value:
            self._value = new_value
            self._notify_dependents()

class ComputeCell(Cell):
    def __init__(self, inputs, compute_function):
        super().__init__()
        self._inputs = inputs
        self._compute_function = compute_function
        self._value = None
        
        # Register this cell as dependent of all inputs
        for input_cell in inputs:
            input_cell._dependents.add(self)
            
    @property
    def value(self):
        if self._value is None:
            self._recompute()
        return self._value
        
    def _recompute(self):
        old_value = self._value
        input_values = [input_cell.value for input_cell in self._inputs]
        self._value = self._compute_function(input_values)
        
        # Notify callbacks if value changed
        if old_value is not None and self._value != old_value:
            self._notify_callbacks()
            
    def _invalidate(self):
        """Override to handle callback notification on value change"""
        old_value = self._value
        super()._invalidate()
        
        # If we had a value before, we need to recompute and check for changes
        if old_value is not None:
            new_value = self.value
            if new_value != old_value:
                self._notify_callbacks()