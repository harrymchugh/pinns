#!/usr/bin/env python3

from time import time
  
def function_timer(function_to_time):
    """
    """
    def wrap_func(*args, **kwargs):
        start_time = time()
        function_output = function_to_time(*args, **kwargs)
        end_time = time()
        print(f"{function_to_time.__name__!r} wall-time (seconds): {(end_time-start_time):.3f}")
        return function_output
    
    return wrap_func
