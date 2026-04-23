from __future__ import annotations


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch