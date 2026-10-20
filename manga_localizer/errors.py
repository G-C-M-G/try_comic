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


class TranslationTimeoutError(TranslationError):
    """A translator request used up its whole timeout budget without answering.

    单独一类是为了**可诊断**：以前所有传输层故障都塌成同一句 "request failed"，
    于是"到底是超时、还是 DNS/连接被重置"在日志里分不出来，也没法据此决定要不要
    重试。已完成的超时意味着本次请求预算已经花光——在同一个引擎上原样再来一次
    大概率再花光一次（参考实现的同一条规则：``InterruptedIOException`` 不重试）。
    """


class TypesettingError(MangaLocalizerError):
    """Chinese text cannot be placed without violating the layout contract."""


class StateError(MangaLocalizerError):
    """An editable state is invalid, stale, or bound to different inputs."""
