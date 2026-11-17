"""Small graph-node pool used by BlindGateGraphBrain."""
from __future__ import annotations
from collections.abc import Hashable

class DynamicQPool:
    """Store learned graph edges and expose pool statistics."""
    def __init__(self, name: str, capacity: int) -> None:
        self.name=name; self.capacity=int(capacity); self.nodes: dict[Hashable,float]={}
        self.seeded_count=0; self.spawned_count=0; self.pruned_count=0
    def get(self,key:Hashable)->float: return float(self.nodes.get(key,0.0))
    def stats(self)->dict[str,int]:
        return {"seeded":self.seeded_count,"spawned":self.spawned_count,"pruned":self.pruned_count,"active":len(self.nodes)}

__all__ = ["DynamicQPool"]
