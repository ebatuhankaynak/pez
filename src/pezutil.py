"""Small shared helpers with no heavy deps — safe to import from the GPU-free stages."""


def short(name):
    """Clip id = last 12 chars of the filename stem. The .mp4 guard keeps an already-short
    id (no extension) from being truncated by the blind [:-4] strip."""
    return name[:-4][-12:] if name.endswith(".mp4") else name
