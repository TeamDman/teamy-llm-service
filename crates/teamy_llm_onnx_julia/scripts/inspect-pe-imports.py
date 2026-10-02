"""Read PE import names from an explicit DLL without loading or executing it."""

import argparse
import json
from pathlib import Path
import struct


def inspect(path):
    data = path.read_bytes()
    if data[:2] != b"MZ":
        raise ValueError("Not a PE image")
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe:pe + 4] != b"PE\0\0":
        raise ValueError("Invalid PE signature")
    machine, count = struct.unpack_from("<HH", data, pe + 4)
    optional_size = struct.unpack_from("<H", data, pe + 20)[0]
    optional = pe + 24
    if struct.unpack_from("<H", data, optional)[0] != 0x20B:
        raise ValueError("Expected a PE32+ image")
    directories = optional + 112
    sections = []
    for index in range(count):
        section = optional + optional_size + index * 40
        virtual_size, rva, raw_size, offset = struct.unpack_from("<IIII", data, section + 8)
        sections.append((rva, max(virtual_size, raw_size), offset, raw_size))

    def offset(rva):
        for start, size, raw, raw_size in sections:
            if start <= rva < start + size and rva - start < raw_size:
                return raw + rva - start
        raise ValueError("Import RVA is outside file-backed sections")

    def name(rva):
        start = offset(rva)
        end = data.index(b"\0", start, start + 512)
        return data[start:end].decode("ascii")

    imports, delayed = [], []
    import_rva, import_size = struct.unpack_from("<II", data, directories + 8)
    if import_rva:
        for entry in range(import_size // 20):
            descriptor = struct.unpack_from("<IIIII", data, offset(import_rva) + entry * 20)
            if not any(descriptor):
                break
            imports.append(name(descriptor[3]))
    delay_rva, delay_size = struct.unpack_from("<II", data, directories + 13 * 8)
    if delay_rva:
        for entry in range(delay_size // 32):
            descriptor = struct.unpack_from("<IIIIIIII", data, offset(delay_rva) + entry * 32)
            if not any(descriptor):
                break
            if descriptor[0] != 1:
                raise ValueError("Delay import uses unsupported virtual-address names")
            delayed.append(name(descriptor[1]))
    return {"file": path.name, "machine": hex(machine),
            "imports": imports, "delay_imports": delayed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dlls", type=Path, nargs="+")
    print(json.dumps([inspect(path.resolve(strict=True)) for path in parser.parse_args().dlls], indent=2))
