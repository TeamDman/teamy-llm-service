"""Verify and inspect/extract a pinned ort archive. Does not load native libraries."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


EXPECTED_SHA256 = "72207a283ec9886e1968a4183636a7665c78e2154d4f4adc16e61193dd7a74f1"
EXPECTED_BYTES = 288348147
MAX_BYTES = 2 * 1024**3
MAX_MEMBERS = 256
DLL_NAMES = {"onnxruntime.dll", "onnxruntime_providers_shared.dll", "onnxruntime_providers_cuda.dll"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--extract-to", type=Path)
    args = parser.parse_args()
    archive = args.archive.resolve(strict=True)
    with archive.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if archive.stat().st_size != EXPECTED_BYTES or digest != EXPECTED_SHA256:
        raise ValueError("Archive does not match the frozen CUDA13 package")
    root = args.extract_to.resolve() if args.extract_to else None
    if root:
        # Acquisition writes are confined to a child beside the explicitly supplied archive.
        if not root.is_relative_to(archive.parent) or root == archive.parent:
            raise ValueError("Extraction must be below the archive's staging directory")
        root.mkdir(parents=True, exist_ok=True)
    records, seen, selected, total = [], set(), set(), 0
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
                name = PurePosixPath(member.filename)
                if (name.is_absolute() or ".." in name.parts or "\\" in member.filename
                        or ":" in member.filename or member.filename in seen):
                    raise ValueError("Unsafe or duplicate archive member")
                seen.add(member.filename)
                unix_type = (member.external_attr >> 16) & 0o170000
                if len(seen) > MAX_MEMBERS or unix_type == 0o120000:
                    raise ValueError("Unsupported archive entry type or count")
                total += member.file_size
                if member.file_size < 0 or total > MAX_BYTES:
                    raise ValueError("Runtime archive exceeds the extraction bound")
                record = {"path": member.filename, "bytes": member.file_size}
                if name.name in DLL_NAMES and not member.is_dir():
                    if name.name in selected:
                        raise ValueError("Duplicate selected runtime library")
                    selected.add(name.name)
                    stream = bundle.open(member)
                    sha256 = hashlib.sha256()
                    destination = None
                    if root:
                        path = root.joinpath(*name.parts)
                        if not path.resolve().is_relative_to(root):
                            raise ValueError("Archive member escapes staging root")
                        path.parent.mkdir(parents=True, exist_ok=True)
                        destination = path.open("xb")
                    try:
                        while data := stream.read(1024 * 1024):
                            sha256.update(data)
                            if destination:
                                destination.write(data)
                    finally:
                        stream.close()
                        if destination:
                            destination.close()
                    record["sha256"] = sha256.hexdigest()
                records.append(record)
    if selected != DLL_NAMES:
        raise ValueError("Archive does not contain every required runtime library")
    print(json.dumps({"version": 1, "archive_sha256": digest,
                      "archive_bytes": EXPECTED_BYTES, "expanded_bytes": total,
                      "extracted": bool(root), "files": records}, indent=2))


if __name__ == "__main__":
    main()
