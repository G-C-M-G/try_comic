from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import time
from typing import Dict, Mapping, Optional, Protocol, Tuple

from PIL import Image, ImageChops, ImageFilter, ImageOps

from .errors import ErasureError
from .ocr_contract import BBox, OCRContract
from .region_masks import (
    ColumnMask,
    ErasureColumnMask,
    ErasureMaskConfig,
    ErasureMaskSet,
    MaskSet,
    build_column_masks,
    build_erasure_masks,
)


RGBColor = Tuple[int, int, int]


@dataclass(frozen=True)
class BackgroundMetrics:
    sample_pixels: int
    mean_luminance: float
    luminance_stddev: float
    white_ratio: float
    dark_ratio: float


@dataclass(frozen=True)
class EraseStyleDecision:
    column_id: str
    strategy: str
    classifier: str
    reason: str
    metrics: BackgroundMetrics


class EraseStyleClassifier(Protocol):
    name: str

    def classify(self, image: Image.Image, geometry: ColumnMask) -> EraseStyleDecision:
        ...


class Inpainter(Protocol):
    name: str

    def inpaint(self, image: Image.Image, mask: Image.Image, padding: int) -> Image.Image:
        ...


@dataclass(frozen=True)
class LocalStatisticsStyleClassifier:
    ring_radius: int = 8
    solid_min_luminance: float = 225.0
    solid_min_white_ratio: float = 0.75
    name: str = "local-background-statistics-v1"

    def classify(self, image: Image.Image, geometry: ColumnMask) -> EraseStyleDecision:
        metrics = measure_background(image, geometry.image, self.ring_radius)
        solid = (
            metrics.mean_luminance >= self.solid_min_luminance
            and metrics.white_ratio >= self.solid_min_white_ratio
        )
        strategy = "solid_fill" if solid else "inpaint"
        reason = (
            "bright-low-complexity-background"
            if solid
            else "line-art-tone-or-complex-background"
        )
        return EraseStyleDecision(
            column_id=geometry.audit.column_id,
            strategy=strategy,
            classifier=self.name,
            reason=reason,
            metrics=metrics,
        )


@dataclass(frozen=True)
class DirectionalInpainter:
    name: str = "directional-linear-local-v1"

    def inpaint(self, image: Image.Image, mask: Image.Image, padding: int) -> Image.Image:
        bbox = mask.getbbox()
        if bbox is None:
            return image.copy()
        box = _expanded_bbox(bbox, max(1, padding), image.size)
        local_image = image.crop(box).convert("RGB")
        local_mask = mask.crop(box).convert("L")
        fallback = sample_ring_color(image, mask, max(2, padding))
        filled = _directional_fill(local_image, local_mask, fallback, blend=True)
        local_result = Image.composite(filled, local_image, local_mask)
        result = image.copy()
        result.paste(local_result, box[:2])
        return result


@dataclass(frozen=True)
class NearestBoundaryInpainter:
    name: str = "nearest-boundary-local-v1"

    def inpaint(self, image: Image.Image, mask: Image.Image, padding: int) -> Image.Image:
        bbox = mask.getbbox()
        if bbox is None:
            return image.copy()
        box = _expanded_bbox(bbox, max(1, padding), image.size)
        local_image = image.crop(box).convert("RGB")
        local_mask = mask.crop(box).convert("L")
        fallback = sample_ring_color(image, mask, max(2, padding))
        filled = _directional_fill(local_image, local_mask, fallback, blend=False)
        local_result = Image.composite(filled, local_image, local_mask)
        result = image.copy()
        result.paste(local_result, box[:2])
        return result


@dataclass(frozen=True)
class OpenCVFSRInpainter:
    name: str = "opencv-fsr-fast-local-v1"

    def inpaint(self, image: Image.Image, mask: Image.Image, padding: int) -> Image.Image:
        bbox = mask.getbbox()
        if bbox is None:
            return image.copy()
        try:
            import cv2
            import numpy as np
        except ImportError as error:
            raise ErasureError(
                "opencv-contrib-python-headless and numpy are required for FSR inpainting"
            ) from error
        box = _expanded_bbox(bbox, max(1, padding), image.size)
        local_image = image.crop(box).convert("RGB")
        local_mask = mask.crop(box).convert("L")
        source = np.asarray(local_image)
        erase_mask = np.asarray(local_mask)
        valid_mask = np.where(erase_mask > 0, 0, 255).astype(np.uint8)
        destination = np.empty_like(source)
        try:
            cv2.xphoto.inpaint(
                source,
                valid_mask,
                destination,
                cv2.xphoto.INPAINT_FSR_FAST,
            )
        except cv2.error as error:
            raise ErasureError(f"OpenCV FSR inpainting failed: {error}") from error
        local_result = Image.fromarray(destination)
        result = image.copy()
        result.paste(local_result, box[:2], local_mask)
        return result


@dataclass(frozen=True)
class ErasureConfig:
    style_override: str = "auto"
    mask: ErasureMaskConfig = ErasureMaskConfig()
    fill_ring_radius: int = 6
    inpaint_padding: int = 8
    # 修复后端：``auto`` = 有 LaMa 模型就用 LaMa、否则 FSR；也可强制 ``lama``/``fsr``。
    # **这个值进擦字缓存键**（`_erase_cache_key`），所以换后端会自动重擦。
    inpainter: str = "auto"


# --- LaMa 修复（2026-09-17 接入）---------------------------------------------
#
# 为什么换掉 FSR：现役 `cv2.xphoto.INPAINT_FSR_FAST` 的选型**项目里从没评估过**。
# 实测对照（同一掩膜、同一批岛，`var/probe_lama_vs_fsr.py`）：
#   * 页2 WARNING 白底：FSR 去掉了字但**留下黑色残块**；LaMa 干净，且保住了原图案的斜杠装饰。
#   * 页1 DOJIN/R18：FSR 把"白带+黑盒"的**盒子结构糊掉**；LaMa 结构保住。
#   * 页16 网点上的手绘字：FSR 把网点抹成**一片模糊白灰**；LaMa **把网点纹理重建出来**。
#   * 速度（生产口径，一页一次）：FSR 随面积暴涨（页22 **40.0s**）；LaMa 恒定 ~1.5s/岛。
#
# 两条**必须守住**的实现约束（我第一版都踩过，结果填出大片纯黑）：
#   1. **上下文要给够**。裁剪紧贴掩膜 → 掩膜占满整张图 → LaMa 没有信息可依 → 瞎猜。
#      这里按组件尺寸的 ``context_ratio`` 倍向外取框。
#   2. **不许硬压方形**。实测 6.8:1 的裁剪压成 512×512 会崩（连 `WARNING` 都去不掉）。
#      这里保长宽比缩放到 512，缺的边用**边框主色**补（中性，不注入结构）。
#
# 调用方传进来的是**整页并集掩膜**（每个后端一页只调一次），所以这里自己拆连通域、
# 逐个给上下文跑。
LAMA_MODEL = (
    Path(__file__).resolve().parents[1] / "models" / "lama-manga" / "lama-manga.onnx"
)
LAMA_INPUT_SIZE = 512            # 这份导出是固定 512×512（无动态 shape）
LAMA_CONTEXT_RATIO = 0.6         # 组件之外各留 0.6 倍尺寸作上下文
LAMA_MIN_COMPONENT_PIXELS = 24    # 太小的组件不值得跑模型


@dataclass(frozen=True)
class LamaInpainter:
    """Big-LaMa（漫画微调）修复：逐连通域、给足上下文、保长宽比。"""

    model_path: Path = LAMA_MODEL
    context_ratio: float = LAMA_CONTEXT_RATIO
    size: int = LAMA_INPUT_SIZE
    name: str = "lama-manga-big-local-v1"

    def available(self) -> bool:
        if not Path(self.model_path).is_file():
            return False
        try:
            import numpy  # noqa: F401
            import onnxruntime  # noqa: F401
            import scipy  # noqa: F401
        except ImportError:
            return False
        return True

    @staticmethod
    def _expose_cuda_runtime_dlls() -> None:
        """把 torch 自带的 CUDA 运行时目录加进 DLL 搜索路径。

        **onnxruntime-gpu 不会自己找 cuDNN / cuBLAS，它靠 PATH。** 装了 CUDA 版 torch
        的环境里这些 DLL 就在 ``<site-packages>/torch/lib``，但不在 PATH 上——实测缺
        ``cublasLt64_12.dll`` 时 CUDA provider **静默退回 CPU**（只有一条 warning，
        很容易被日志淹没，看起来"用了 GPU"其实没有）。所以这里显式加进去。

        另外注意**版本要对**：onnxruntime-gpu 1.30 要 CUDA 13（``cublasLt64_13.dll``），
        而 torch cu126 带的是 CUDA 12 命名 → 对不上。本机可用组合是
        **onnxruntime-gpu 1.22.0**（CUDA 12 构建）+ torch 2.13+cu126 的 DLL。
        """
        if os.name != "nt":
            return
        try:
            import torch
        except ImportError:
            return
        library = Path(torch.__file__).resolve().parent / "lib"
        if not library.is_dir():
            return
        os.environ["PATH"] = str(library) + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(str(library))
        except (AttributeError, OSError):
            pass

    def _session(self):
        cached = getattr(self, "_cached_session", None)
        if cached is None:
            import onnxruntime

            available = onnxruntime.get_available_providers()
            providers: list[str] = []
            if "CUDAExecutionProvider" in available:
                self._expose_cuda_runtime_dlls()
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")
            options = onnxruntime.SessionOptions()
            # **限制 CPU 路每路会话的线程数**：页级并行（`--max-workers` 默认 3）已经
            # 把多个 LaMa 会话同时放进来了，而 onnxruntime 默认每路都吃满所有核 →
            # 线程超订、互相抢。实测（16 逻辑核 / 3 路 / CPU）不限制时 32s/页，
            # 而单页单独跑只要 38s（3 路本该 ~13s/页）——并行几乎白拿。
            # GPU 路不吃 CPU 线程，这个设置对它无害。
            options.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 3)
            options.inter_op_num_threads = 1
            cached = onnxruntime.InferenceSession(
                str(self.model_path), options, providers=providers)
            object.__setattr__(self, "_cached_session", cached)
        return cached

    def _fill_component(self, working: Image.Image, box: BBox,
                        component: Image.Image) -> None:
        import numpy as np

        crop = working.crop(box)
        component_crop = component.crop(box)
        width, height = crop.size
        scale = self.size / max(width, height)
        small_w = max(1, round(width * scale))
        small_h = max(1, round(height * scale))
        # 缺的边用**边框主色**补，保持中性（不用白/黑，免得给模型暗示）
        edge = crop.resize((1, 1), Image.BOX).getpixel((0, 0))
        canvas = Image.new("RGB", (self.size, self.size), edge)
        canvas.paste(crop.resize((small_w, small_h), Image.LANCZOS), (0, 0))
        mask_canvas = Image.new("L", (self.size, self.size), 0)
        mask_canvas.paste(
            component_crop.resize((small_w, small_h), Image.NEAREST), (0, 0))

        image_input = np.transpose(
            np.asarray(canvas, dtype=np.float32) / 255.0, (2, 0, 1))[None, ...]
        mask_input = (
            np.asarray(mask_canvas, dtype=np.float32) / 255.0)[None, None, ...]
        output = self._session().run(
            ["output"], {"image": image_input, "mask": mask_input})[0][0]
        filled = Image.fromarray(
            np.clip(np.transpose(output, (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)
        ).crop((0, 0, small_w, small_h)).resize((width, height), Image.LANCZOS)
        local = crop.copy()
        local.paste(filled, (0, 0), component_crop)
        working.paste(local, (box[0], box[1]))

    def inpaint(self, image: Image.Image, mask: Image.Image, padding: int) -> Image.Image:
        if mask.getbbox() is None:
            return image.copy()
        try:
            import numpy as np
            from scipy import ndimage
        except ImportError as error:
            raise ErasureError(
                "numpy 与 scipy 是 LaMa 修复的依赖（pip install scipy）"
            ) from error

        source = image.convert("RGB")
        binary = np.asarray(mask.convert("L")) > 0
        labels, count = ndimage.label(binary)
        if count == 0:
            return source.copy()
        working = source.copy()
        for index in range(1, count + 1):
            rows, columns = np.nonzero(labels == index)
            if rows.size < LAMA_MIN_COMPONENT_PIXELS:
                continue
            y1, y2 = int(rows.min()), int(rows.max()) + 1
            x1, x2 = int(columns.min()), int(columns.max()) + 1
            margin = int(round(max(x2 - x1, y2 - y1) * self.context_ratio))
            box = (max(0, x1 - margin), max(0, y1 - margin),
                   min(source.width, x2 + margin), min(source.height, y2 + margin))
            component = Image.fromarray(
                np.where(labels == index, 255, 0).astype(np.uint8))
            self._fill_component(working, box, component)
        # 只保留掩膜内的改动——掩膜外的像素必须逐位不变（擦字的像素完整性契约）
        return Image.composite(working, source, mask.convert("L"))


def select_inpainter(config: Optional["ErasureConfig"] = None) -> "Inpainter":
    """按配置挑修复后端；``auto`` 在 LaMa 可用时优先用它，否则退回 FSR。

    可用性在**开跑前**判定（模型文件 + 依赖），不会跑到第 80 页才发现模型缺失。
    """
    choice = str(getattr(config, "inpainter", "auto") or "auto").lower()
    if choice in ("fsr", "opencv"):
        return OpenCVFSRInpainter()
    lama = LamaInpainter()
    if choice == "lama":
        if not lama.available():
            raise ErasureError(
                f"配置要求 LaMa 修复，但模型不可用：{lama.model_path}"
                "（见 开发记录 §13；或把 inpainter 改回 \"fsr\"）"
            )
        return lama
    return lama if lama.available() else OpenCVFSRInpainter()


@dataclass(frozen=True)
class ColumnErasureAudit:
    column_id: str
    unit_id: str
    reading_position: int
    strategy: str
    classifier: str
    classifier_reason: str
    background: BackgroundMetrics
    mask_kind: str
    mask_bbox: BBox
    mask_pixels: int
    geometry_pixels: int
    threshold: Optional[int]
    dilation_pixels: int
    contains_geometry_envelope: bool
    white_outline_pixels: int
    backend: str
    fill_color: Optional[RGBColor]
    changed_pixels: int


@dataclass(frozen=True)
class PixelDifferenceAudit:
    allowed_mask_pixels: int
    changed_pixels: int
    changed_inside_mask_pixels: int
    changed_outside_mask_pixels: int
    unchanged_mask_pixels: int
    outside_mask_unchanged: bool
    difference_bbox: Optional[BBox]
    max_channel_delta: int
    sum_absolute_difference: int


@dataclass(frozen=True)
class ErasureResult:
    image: Image.Image
    difference: Image.Image
    changed_mask: Image.Image
    geometry_masks: MaskSet
    erasure_masks: ErasureMaskSet
    decisions: Mapping[str, EraseStyleDecision]
    columns: Tuple[ColumnErasureAudit, ...]
    pixels: PixelDifferenceAudit
    erasure_seconds: float
    pixel_audit_seconds: float


def erase_image(
    image: Image.Image,
    contract: OCRContract,
    *,
    config: ErasureConfig = ErasureConfig(),
    classifier: Optional[EraseStyleClassifier] = None,
    inpainter: Optional[Inpainter] = None,
) -> ErasureResult:
    erasure_started = time.perf_counter()
    if image.size != contract.image_size:
        raise ErasureError(
            f"image size {image.size} differs from OCR contract {contract.image_size}"
        )
    if config.style_override not in {"auto", "solid_fill", "inpaint"}:
        raise ErasureError(f"unsupported erase strategy: {config.style_override}")
    if config.fill_ring_radius < 1 or config.inpaint_padding < 1:
        raise ErasureError("erasure sampling and inpaint radii must be positive")

    source = image.convert("RGB")
    geometry_masks = build_column_masks(contract)
    active_classifier = classifier or LocalStatisticsStyleClassifier()
    decisions: Dict[str, EraseStyleDecision] = {}
    for geometry in geometry_masks.columns:
        automatic = active_classifier.classify(source, geometry)
        if config.style_override == "auto":
            decision = automatic
        else:
            decision = EraseStyleDecision(
                column_id=automatic.column_id,
                strategy=config.style_override,
                classifier="cli-override",
                reason=f"forced-{config.style_override}",
                metrics=automatic.metrics,
            )
        decisions[decision.column_id] = decision

    erasure_masks = build_erasure_masks(
        source,
        geometry_masks,
        {column_id: decision.strategy for column_id, decision in decisions.items()},
        config.mask,
    )
    active_inpainter = inpainter or select_inpainter(config)
    working = source.copy()
    execution: Dict[str, Tuple[str, Optional[RGBColor]]] = {}
    inpaint_groups: Dict[str, Tuple[Inpainter, Image.Image]] = {}
    for column_mask in erasure_masks.columns:
        decision = decisions[column_mask.audit.column_id]
        if decision.strategy == "solid_fill":
            color = sample_ring_color(
                source,
                column_mask.image,
                config.fill_ring_radius,
            )
            working.paste(color, (0, 0, working.width, working.height), column_mask.image)
            execution[column_mask.audit.column_id] = ("restricted-ring-median-fill-v1", color)
        else:
            backend = active_inpainter
            group = inpaint_groups.get(backend.name)
            if group is None:
                group = (backend, Image.new("L", source.size, 0))
            inpaint_groups[backend.name] = (
                group[0],
                ImageChops.lighter(group[1], column_mask.image),
            )
            execution[column_mask.audit.column_id] = (backend.name, None)
    for backend, grouped_mask in inpaint_groups.values():
        if grouped_mask.getbbox() is not None:
            working = backend.inpaint(working, grouped_mask, config.inpaint_padding)

    # This final composite is the hard pixel-integrity boundary. Even if a future
    # backend touches a larger crop internally, pixels outside the audited union
    # cannot reach the output.
    # Keep the pixel-integrity audit on its own timer. Erasure and future
    # typesetting work may be expensive, but only this verification budget is
    # used for the one-minute gate.
    pixel_audit_started = time.perf_counter()
    erased = Image.composite(working, source, erasure_masks.combined)
    difference = ImageChops.difference(source, erased)
    changed_mask = _changed_pixel_mask(difference)
    pixels = _pixel_difference_audit(difference, changed_mask, erasure_masks.combined)
    pixel_audit_seconds = time.perf_counter() - pixel_audit_started
    if not pixels.outside_mask_unchanged:
        raise ErasureError(
            f"eraser changed {pixels.changed_outside_mask_pixels} pixels outside the mask"
        )

    mask_map = {item.audit.column_id: item for item in erasure_masks.columns}
    column_audits = []
    for geometry in geometry_masks.columns:
        column_id = geometry.audit.column_id
        column_mask = mask_map[column_id]
        decision = decisions[column_id]
        backend, fill_color = execution[column_id]
        changed = ImageChops.darker(changed_mask, column_mask.image).histogram()[255]
        column_audits.append(
            ColumnErasureAudit(
                column_id=column_id,
                unit_id=column_mask.audit.unit_id,
                reading_position=column_mask.audit.reading_position,
                strategy=decision.strategy,
                classifier=decision.classifier,
                classifier_reason=decision.reason,
                background=decision.metrics,
                mask_kind=column_mask.audit.kind,
                mask_bbox=column_mask.audit.mask_bbox,
                mask_pixels=column_mask.audit.pixel_count,
                geometry_pixels=column_mask.audit.geometry_pixel_count,
                threshold=column_mask.audit.threshold,
                dilation_pixels=column_mask.audit.dilation_pixels,
                contains_geometry_envelope=column_mask.audit.contains_geometry_envelope,
                white_outline_pixels=column_mask.audit.white_outline_pixels,
                backend=backend,
                fill_color=fill_color,
                changed_pixels=changed,
            )
        )
    return ErasureResult(
        image=erased,
        difference=difference,
        changed_mask=changed_mask,
        geometry_masks=geometry_masks,
        erasure_masks=erasure_masks,
        decisions=decisions,
        columns=tuple(column_audits),
        pixels=pixels,
        erasure_seconds=time.perf_counter() - erasure_started,
        pixel_audit_seconds=pixel_audit_seconds,
    )


def measure_background(
    image: Image.Image,
    mask: Image.Image,
    ring_radius: int,
) -> BackgroundMetrics:
    gray = image.convert("L")
    ring = _mask_ring(mask, max(1, ring_radius))
    histogram = gray.histogram(mask=ring)
    sample_pixels = sum(histogram)
    if not sample_pixels:
        return BackgroundMetrics(0, 255.0, 0.0, 1.0, 0.0)
    mean = sum(value * count for value, count in enumerate(histogram)) / sample_pixels
    variance = (
        sum(((value - mean) ** 2) * count for value, count in enumerate(histogram))
        / sample_pixels
    )
    return BackgroundMetrics(
        sample_pixels=sample_pixels,
        mean_luminance=mean,
        luminance_stddev=math.sqrt(variance),
        white_ratio=sum(histogram[240:]) / sample_pixels,
        dark_ratio=sum(histogram[:80]) / sample_pixels,
    )


def sample_ring_color(image: Image.Image, mask: Image.Image, radius: int) -> RGBColor:
    rgb = image.convert("RGB")
    ring = _mask_ring(mask, max(1, radius))
    return tuple(_histogram_median(band.histogram(mask=ring), 255) for band in rgb.split())  # type: ignore[return-value]


def _directional_fill(
    image: Image.Image,
    mask: Image.Image,
    fallback: RGBColor,
    *,
    blend: bool,
) -> Image.Image:
    width, height = image.size
    pixels = _flattened_pixels(image)
    masked = [value > 0 for value in _flattened_pixels(mask)]
    red = [0.0] * len(pixels)
    green = [0.0] * len(pixels)
    blue = [0.0] * len(pixels)
    weights = [0.0] * len(pixels)

    def add(index: int, color: RGBColor, weight: float) -> None:
        red[index] += color[0] * weight
        green[index] += color[1] * weight
        blue[index] += color[2] * weight
        weights[index] += weight

    for y in range(height):
        x = 0
        while x < width:
            index = y * width + x
            if not masked[index]:
                x += 1
                continue
            start = x
            while x < width and masked[y * width + x]:
                x += 1
            end = x
            left = pixels[y * width + start - 1] if start > 0 else None
            right = pixels[y * width + end] if end < width else None
            span = end - start
            weight = 1.0 / max(1, span)
            for offset, current_x in enumerate(range(start, end), 1):
                if left is not None and right is not None:
                    fraction = offset / (span + 1)
                    color = (
                        tuple(
                            round(left[channel] * (1.0 - fraction) + right[channel] * fraction)
                            for channel in range(3)
                        )
                        if blend
                        else left
                        if offset * 2 <= span + 1
                        else right
                    )
                else:
                    color = left or right or fallback
                add(y * width + current_x, color, weight)  # type: ignore[arg-type]

    for x in range(width):
        y = 0
        while y < height:
            index = y * width + x
            if not masked[index]:
                y += 1
                continue
            start = y
            while y < height and masked[y * width + x]:
                y += 1
            end = y
            top = pixels[(start - 1) * width + x] if start > 0 else None
            bottom = pixels[end * width + x] if end < height else None
            span = end - start
            weight = 1.0 / max(1, span)
            for offset, current_y in enumerate(range(start, end), 1):
                if top is not None and bottom is not None:
                    fraction = offset / (span + 1)
                    color = (
                        tuple(
                            round(top[channel] * (1.0 - fraction) + bottom[channel] * fraction)
                            for channel in range(3)
                        )
                        if blend
                        else top
                        if offset * 2 <= span + 1
                        else bottom
                    )
                else:
                    color = top or bottom or fallback
                add(current_y * width + x, color, weight)  # type: ignore[arg-type]

    output = list(pixels)
    for index, is_masked in enumerate(masked):
        if not is_masked:
            continue
        if weights[index]:
            output[index] = (
                round(red[index] / weights[index]),
                round(green[index] / weights[index]),
                round(blue[index] / weights[index]),
            )
        else:
            output[index] = fallback
    result = Image.new("RGB", image.size)
    result.putdata(output)
    return result


def _pixel_difference_audit(
    difference: Image.Image,
    changed_mask: Image.Image,
    allowed_mask: Image.Image,
) -> PixelDifferenceAudit:
    outside = ImageOps.invert(allowed_mask)
    changed_inside = ImageChops.darker(changed_mask, allowed_mask).histogram()[255]
    changed_outside = ImageChops.darker(changed_mask, outside).histogram()[255]
    changed_pixels = changed_mask.histogram()[255]
    allowed_pixels = allowed_mask.histogram()[255]
    bbox = changed_mask.getbbox()
    histograms = [band.histogram() for band in difference.convert("RGB").split()]
    max_delta = max(
        (value for histogram in histograms for value, count in enumerate(histogram) if count),
        default=0,
    )
    absolute_sum = sum(
        value * count for histogram in histograms for value, count in enumerate(histogram)
    )
    return PixelDifferenceAudit(
        allowed_mask_pixels=allowed_pixels,
        changed_pixels=changed_pixels,
        changed_inside_mask_pixels=changed_inside,
        changed_outside_mask_pixels=changed_outside,
        unchanged_mask_pixels=allowed_pixels - changed_inside,
        outside_mask_unchanged=changed_outside == 0,
        difference_bbox=(
            tuple(int(value) for value in bbox) if bbox is not None else None  # type: ignore[arg-type]
        ),
        max_channel_delta=max_delta,
        sum_absolute_difference=absolute_sum,
    )


def _changed_pixel_mask(difference: Image.Image) -> Image.Image:
    bands = [band.point(lambda value: 255 if value else 0, mode="L") for band in difference.split()]
    changed = bands[0]
    for band in bands[1:]:
        changed = ImageChops.lighter(changed, band)
    return changed


def _mask_ring(mask: Image.Image, radius: int) -> Image.Image:
    bbox = mask.getbbox()
    if bbox is None:
        return Image.new("L", mask.size, 0)
    x1, y1, x2, y2 = bbox
    box = _expanded_bbox(bbox, radius, mask.size)
    local = mask.crop(box).filter(ImageFilter.MaxFilter(radius * 2 + 1))
    dilated = Image.new("L", mask.size, 0)
    dilated.paste(local, box[:2])
    return ImageChops.subtract(dilated, mask)


def _expanded_bbox(bbox: BBox, padding: int, size: Tuple[int, int]) -> BBox:
    x1, y1, x2, y2 = bbox
    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(size[0], x2 + padding),
        min(size[1], y2 + padding),
    )


def _histogram_median(histogram: list[int], default: int) -> int:
    total = sum(histogram)
    if not total:
        return default
    midpoint = (total + 1) // 2
    running = 0
    for value, count in enumerate(histogram):
        running += count
        if running >= midpoint:
            return value
    return default


def _flattened_pixels(image: Image.Image):
    flattened = getattr(image, "get_flattened_data", None)
    return list(flattened()) if flattened is not None else list(image.getdata())
