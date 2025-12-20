from pathlib import Path

from app.services.logger import logger


def dragEnterEvent(self, event):
    if not event.mimeData().hasUrls():
        return

    for url in event.mimeData().urls():
        suffix = Path(url.toLocalFile()).suffix.lower()
        if suffix in {
            ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif", ".jfif",
            ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".3gp",
        }:
            event.acceptProposedAction()
            return


def dropEvent(self, event):
    for url in event.mimeData().urls():
        path = Path(url.toLocalFile())
        suffix = path.suffix.lower()

        image_exts = {
            ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif", ".jfif"
        }
        video_exts = {
            ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".3gp"
        }

        if suffix in image_exts:
            logger.info(f"Перетащено изображение: {path}")
            self.load_image_from_path(path)
            return

        if suffix in video_exts:
            logger.info(f"Перетащено видео: {path}")
            self._load_video(path)
            return
