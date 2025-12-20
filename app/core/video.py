# app/core/video.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import math
import tempfile
import subprocess

from PIL import Image as PILImage

from app.services.logger import logger


VIDEO_EXTS = {
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".3gp"
}


def is_video_path(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTS


@dataclass(frozen=True)
class VideoMeta:
    duration_sec: float
    fps: float
    total_frames: int


def _pick_num_samples(duration_sec: float) -> int:
    if duration_sec <= 10:
        return 24
    if duration_sec <= 60:
        return 48
    if duration_sec <= 300:
        return 96
    return 128


def _run_ffmpeg_transcode_to_mp4(src: Path) -> Path:
    """
    Фоллбек: приводим к MP4 (H.264, yuv420p), чтобы декодирование было максимально совместимым.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="deepfake_video_"))
    dst = tmp_dir / f"{src.stem}_normalized.mp4"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(dst),
    ]

    logger.info(f"FFmpeg transcode: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except Exception as e:
        logger.error(f"FFmpeg transcode failed: {e}")
        raise

    return dst


def read_video_uniform_frames(
    path: Path,
    max_side: int = 768,
    prefer_pyav: bool = True,
    allow_ffmpeg_fallback: bool = True,
) -> Tuple[List[PILImage.Image], VideoMeta]:
    """
    Декодируем видео и возвращаем N PIL-кадров, равномерно по длительности.
    max_side: ограничение на размер кадра для ускорения.
    """
    if prefer_pyav:
        try:
            return _read_with_pyav(path, max_side=max_side)
        except Exception as e:
            logger.warning(f"PyAV read failed, fallback to OpenCV. Reason: {e}")

    try:
        return _read_with_opencv(path, max_side=max_side)
    except Exception as e:
        logger.warning(f"OpenCV read failed. Reason: {e}")

    if allow_ffmpeg_fallback:
        normalized = _run_ffmpeg_transcode_to_mp4(path)
        # после перекодировки пробуем снова
        if prefer_pyav:
            try:
                return _read_with_pyav(normalized, max_side=max_side)
            except Exception as e:
                logger.warning(f"PyAV read after ffmpeg failed. Reason: {e}")
        return _read_with_opencv(normalized, max_side=max_side)

    raise RuntimeError("Unable to decode video with available backends.")


def _resize_keep_aspect(w: int, h: int, max_side: int) -> Tuple[int, int]:
    if max(w, h) <= max_side:
        return w, h
    scale = max_side / float(max(w, h))
    return int(round(w * scale)), int(round(h * scale))


def _read_with_opencv(path: Path, max_side: int) -> Tuple[List[PILImage.Image], VideoMeta]:
    import cv2  # type: ignore

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0.0 and total_frames > 0:
        # fallback: оценка fps (не всегда корректно)
        fps = 25.0
    duration_sec = (total_frames / fps) if (fps > 0 and total_frames > 0) else 0.0

    if duration_sec <= 0:
        # пытаемся "прочитать" часть чтобы оценить
        duration_sec = max(1.0, total_frames / max(fps, 1.0))

    n_samples = min(_pick_num_samples(duration_sec), max(total_frames, 1))
    idxs = _uniform_indices(total_frames, n_samples)

    frames: List[PILImage.Image] = []
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    new_w, new_h = _resize_keep_aspect(w, h, max_side) if w and h else (0, 0)

    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, bgr = cap.read()
        if not ok or bgr is None:
            continue
        if new_w and new_h and (new_w != bgr.shape[1] or new_h != bgr.shape[0]):
            bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(PILImage.fromarray(rgb))

    cap.release()

    meta = VideoMeta(duration_sec=float(duration_sec), fps=float(fps), total_frames=int(total_frames))
    logger.info(f"Video meta (OpenCV): {meta}, sampled_frames={len(frames)}")
    return frames, meta


def _read_with_pyav(path: Path, max_side: int) -> Tuple[List[PILImage.Image], VideoMeta]:
    import av  # type: ignore

    container = av.open(str(path))
    stream = next((s for s in container.streams if s.type == "video"), None)
    if stream is None:
        raise RuntimeError("No video stream found.")

    # fps может быть None
    fps = float(stream.average_rate) if stream.average_rate is not None else 0.0
    duration_sec = float(stream.duration * stream.time_base) if stream.duration else 0.0

    # total_frames может быть неизвестно
    total_frames = int(stream.frames) if stream.frames else 0
    if duration_sec <= 0.0 and total_frames > 0 and fps > 0.0:
        duration_sec = total_frames / fps
    if duration_sec <= 0.0:
        # хотя бы что-то
        duration_sec = 10.0

    n_samples = _pick_num_samples(duration_sec)

    # Семплинг по timestamp: выбираем целевые времена и извлекаем ближайшие кадры.
    target_ts = [duration_sec * (k + 0.5) / n_samples for k in range(n_samples)]

    frames: List[PILImage.Image] = []
    next_target_i = 0

    for frame in container.decode(video=0):
        if next_target_i >= len(target_ts):
            break
        t = float(frame.pts * frame.time_base) if frame.pts is not None else None
        if t is None:
            continue
        if t < target_ts[next_target_i]:
            continue

        img = frame.to_image()  # PIL Image
        w, h = img.size
        new_w, new_h = _resize_keep_aspect(w, h, max_side)
        if (new_w, new_h) != (w, h):
            img = img.resize((new_w, new_h))
        frames.append(img)

        next_target_i += 1

    container.close()

    meta = VideoMeta(duration_sec=float(duration_sec), fps=float(fps), total_frames=int(total_frames))
    logger.info(f"Video meta (PyAV): {meta}, sampled_frames={len(frames)}")
    return frames, meta


def _uniform_indices(total_frames: int, n_samples: int) -> List[int]:
    if total_frames <= 0:
        return [0] * n_samples
    if n_samples <= 1:
        return [total_frames // 2]
    step = total_frames / float(n_samples)
    return [min(total_frames - 1, int((k + 0.5) * step)) for k in range(n_samples)]
