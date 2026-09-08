#!/usr/bin/env python3
"""修补 Mach-O 零填充段的 offset 字段（macOS 26+ dyld 严格校验）.

新版 macOS 的 dyld 要求零填充段（S_ZEROFILL / S_GB_ZEROFILL /
S_THREAD_LOCAL_ZEROFILL）的 offset 字段必须为 0。旧工具链（如
gfortran）编译出的二进制（scipy 的 Fortran 扩展、opencv 捆绑的
部分 dylib 等）携带非零的残留 offset，加载时被拒绝：

    dlopen(...): section '__DATA/__thread_bss' has a zero-fill
    section type, but offset field is not zero

零填充段的内容从不来自文件，因此把 offset 归零是安全的。
必须在 codesign 之前运行（修补会改动 __TEXT 中的 load command）。

用法: python3 patch_zerofill.py <path-to-.app-or-dir>
"""
import struct
import sys
from pathlib import Path

MH_MAGIC_64 = 0xFEEDFACF          # 64 位 Mach-O（小端）
FAT_MAGIC = 0xCAFEBABE            # universal binary
FAT_MAGIC_64 = 0xCAFEBABF
LC_SEGMENT_64 = 0x19
ZEROFILL_TYPES = {0x1, 0xC, 0x12}  # S_ZEROFILL / S_GB_ZEROFILL / S_THREAD_LOCAL_ZEROFILL


def patch_thin(buf: bytearray, base: int) -> list:
    """修补一个 64 位 Mach-O slice，返回 [(seg, sect, old_offset)]。"""
    patched = []
    ncmds = struct.unpack_from("<I", buf, base + 16)[0]
    off = base + 32  # mach_header_64 大小
    for _ in range(ncmds):
        cmd, cmdsize = struct.unpack_from("<II", buf, off)
        if cmd == LC_SEGMENT_64:
            segname = buf[off + 8:off + 24].rstrip(b"\0").decode(errors="replace")
            nsects = struct.unpack_from("<I", buf, off + 64)[0]
            for i in range(nsects):
                s = off + 72 + i * 80  # section_64
                sectname = buf[s:s + 16].rstrip(b"\0").decode(errors="replace")
                sect_off_pos = s + 48
                sect_off = struct.unpack_from("<I", buf, sect_off_pos)[0]
                flags = struct.unpack_from("<I", buf, s + 64)[0]
                if (flags & 0xFF) in ZEROFILL_TYPES and sect_off != 0:
                    struct.pack_into("<I", buf, sect_off_pos, 0)
                    patched.append((segname, sectname, sect_off))
        off += cmdsize
    return patched


def patch_macho(path: Path) -> list:
    """修补单个 Mach-O 文件（支持 universal），返回修补记录。"""
    buf = bytearray(path.read_bytes())
    if len(buf) < 32:
        return []
    results = []
    magic_be = struct.unpack_from(">I", buf, 0)[0]
    if magic_be in (FAT_MAGIC, FAT_MAGIC_64):
        nfat = struct.unpack_from(">I", buf, 4)[0]
        is64 = magic_be == FAT_MAGIC_64
        rec = 32 if is64 else 20
        for i in range(nfat):
            o = 8 + i * rec
            slice_off = struct.unpack_from(">Q" if is64 else ">I", buf, o + 8)[0]
            results += patch_thin(buf, slice_off)
    elif struct.unpack_from("<I", buf, 0)[0] == MH_MAGIC_64:
        results = patch_thin(buf, 0)
    else:
        return []  # 非 Mach-O
    if results:
        path.write_bytes(buf)
    return results


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    root = Path(sys.argv[1])
    if not root.exists():
        print(f"ERROR: {root} 不存在", file=sys.stderr)
        sys.exit(2)

    total = 0
    for p in sorted(root.rglob("*")):
        if p.is_symlink() or not p.is_file():
            continue
        try:
            hits = patch_macho(p)
        except (struct.error, OSError) as e:
            print(f"ERROR: 解析 {p} 失败: {e}", file=sys.stderr)
            sys.exit(1)
        for seg, sect, old in hits:
            print(f"patched {p}: {seg}/{sect} offset 0x{old:x} -> 0")
        total += len(hits)

    # 复扫确认全部清零
    residual = 0
    for p in sorted(root.rglob("*")):
        if p.is_symlink() or not p.is_file():
            continue
        before = p.stat().st_mtime_ns
        hits = patch_macho(p)  # 幂等：干净文件不会被修改
        residual += len(hits)
    if residual:
        print(f"ERROR: 仍有 {residual} 处未修补", file=sys.stderr)
        sys.exit(1)
    print(f"OK: 共修补 {total} 处零填充段 offset，复扫全部合规")


if __name__ == "__main__":
    main()
