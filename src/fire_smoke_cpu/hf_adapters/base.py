from typing import Iterable, Dict, Any

class BaseAdapter:
    def __init__(self, raw_dir: str):
        self.raw_dir = raw_dir
        
    def iter_samples(self) -> Iterable[Dict[str, Any]]:
        raise NotImplementedError("Subclasses must implement iter_samples")
