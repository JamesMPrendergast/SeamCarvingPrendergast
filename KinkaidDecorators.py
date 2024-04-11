import logging
from typing import Callable, Any
import functools
import traceback
import time
"""
Based on an example by Arjan of arjancodes.com: https://www.youtube.com/watch?v=QH5fw9kxDQA 
"""
def log_start_stop_method(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        indent = "\t"*max(0, len(traceback.extract_stack())-1)
        start_time = time.time()
        logging.info(f"{indent}Starting method: {func.__name__}({args}) at {start_time}")
        value = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"{indent}Finishing method: {func.__name__} at {end_time}\t\tit took {elapsed_time}\n")
        return value
    return wrapper