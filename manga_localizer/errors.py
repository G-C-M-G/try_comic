class MangaLocalizerError(Exception):
    """Base error for expected pipeline failures."""


class ContractValidationError(MangaLocalizerError):
    """The OCR JSON violates schema 4 or a cross-field invariant."""


class ImageResolutionError(MangaLocalizerError):
    """No matching source image could be resolved."""


class ImageDimensionError(MangaLocalizerError):
    """The source image dimensions do not match the OCR contract."""


class ReviewValidationError(MangaLocalizerError):
    """A review decision file is invalid or belongs to another OCR input."""


class ReviewBlockedError(MangaLocalizerError):
    """Pending manual review prevents translation."""


class LayoutBoundaryError(MangaLocalizerError):
    """A requested layout boundary is invalid or exceeds the hard ceiling."""


class ErasureError(MangaLocalizerError):
    """The erasure stage cannot preserve its mask or pixel-integrity contract."""


class TranslationError(MangaLocalizerError):
    """A translator request, response, cache entry, or mapping is invalid."""


class TranslationResponseError(TranslationError):
    """A translator returned empty, unchanged, or structurally invalid text."""


class TypesettingError(MangaLocalizerError):
    """Chinese text cannot be placed without violating the layout contract."""


class StateError(MangaLocalizerError):
    """An editable state is invalid, stale, or bound to different inputs."""
