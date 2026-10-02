# Makepad loader source

Source: <https://github.com/makepad/makepad/tree/9e5e3d2b03214f8c2c37adffa0e308c815662060/libs/ai/loader>

Revision: `9e5e3d2b03214f8c2c37adffa0e308c815662060`.

The source, original tests and fixture data were copied from this pinned
revision. The upstream root MIT license is retained in `LICENSE`, including
its copyright and permission notice. The oracle fixture is retained because
the existing quantization tests compile it with `include_bytes!`.
Text uses LF line endings, matching the upstream Git blobs rather than the
Windows checkout's CRLF conversion. Binary fixture bytes are unchanged.

Local changes:

- The manifest declares the upstream MIT license explicitly and resolves
  unchanged `makepad-micro-serde` from the same exact upstream Git revision.
- Windows uses the service's frozen `memmap2` version `0.9.11` and a read-only whole-file mapping.
  The mapping retains an owned `std::fs::File` with `FILE_SHARE_READ`, denying
  ordinary file writes, truncation, deletion and replacement until the last
  mapping owner drops. It exposes the exact file length and rejects empty
  files. Windows working-set residency remains unmeasured: `(0, 0)`.
- Windows fixture tests cover binary bytes, non-page lengths, empty and
  missing files, file sharing, drop lifetime and thread ownership.

The Unix implementation and all format/quantization/numerical code are
unchanged. This mapping does not prevent privileged modification or mutation
through a pre-existing writable mapping. The unsafe mapping contract requires
immutable backing bytes throughout use of borrowed slices. This patch makes
no startup-time, GPU-transfer or numerical-parity claim before qualification.

Cargo dependency activation and validation are managed by the service
workspace. No upstream checkout is modified by this source copy.
