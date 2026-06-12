from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedInfo:  # should not export from here
    rank: int
    size: int

    def __post_init__(self):
        assert 0 <= self.rank < self.size

    def is_primary(self) -> bool:
        return self.rank == 0


_TP_INFO: DistributedInfo | None = None


def set_tp_info(rank: int, size: int) -> None:
    global _TP_INFO
    if _TP_INFO is not None:
        raise RuntimeError("TP info has been set")
    _TP_INFO = DistributedInfo(rank, size)


def get_tp_info() -> DistributedInfo:
    if _TP_INFO is None:
        raise RuntimeError("TP info has not been set")
    return _TP_INFO


def try_get_tp_info() -> DistributedInfo | None:
    return _TP_INFO


_CP_INFO: DistributedInfo | None = None


def set_cp_info(rank: int, size: int) -> None:
    global _CP_INFO
    if _CP_INFO is not None:
        raise RuntimeError("CP info has been set")
    _CP_INFO = DistributedInfo(rank, size)


def get_cp_info() -> DistributedInfo:
    if _CP_INFO is None:
        raise RuntimeError("CP info has not been set")
    return _CP_INFO


def try_get_cp_info() -> DistributedInfo | None:
    return _CP_INFO


__all__ = [
    "DistributedInfo",
    "set_tp_info",
    "get_tp_info",
    "try_get_tp_info",
    "set_cp_info",
    "get_cp_info",
    "try_get_cp_info",
]
