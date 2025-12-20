from app.core.inference import PredictionResult, VideoPredictionResult
from app.services.logger import logger


def run_prediction(self):
    if self.current_media_path is None:
        return

    logger.info("Пользователь запустил анализ.")
    threshold = self.threshold_spin.value() / 100
    self.status_bar.showMessage("Анализ...")

    try:
        if self.current_media_type == "video":
            result = self.classifier.predict_video(
                self.current_media_path,
                threshold=threshold,
                agg_method="median_of_means",
                chunk_count=8,
                batch_size=16,
                max_side=768,
            )
            self._display_video_result(result)
        else:
            if self.current_pil_image is None:
                return
            result = self.classifier.predict(self.current_pil_image, threshold)
            self._display_result(result)

    except Exception as e:
        logger.error(f"Ошибка инференса: {e}")
        self.status_bar.showMessage("Ошибка анализа!")
        return


def _display_result(self, result: PredictionResult):
    prob_pct = int(result.prob_deepfake * 100)
    label_ru = "ДИПФЕЙК" if result.label == "deepfake" else "НАСТОЯЩЕЕ"

    self.result_label.setText(
        f"<b>{label_ru}</b><br>"
        f"Вероятность дипфейка: <b>{result.prob_deepfake:.4f}</b> ({prob_pct}%)"
    )

    self.status_bar.showMessage("Готово.")
    logger.info("Вывод результата завершён.")


def _display_video_result(self, result: VideoPredictionResult):
    prob_pct = int(result.prob_deepfake * 100)
    label_ru = "ДИПФЕЙК" if result.label == "deepfake" else "НАСТОЯЩЕЕ"

    self.result_label.setText(
        f"<b>{label_ru}</b><br>"
        f"Итоговая вероятность дипфейка: <b>{result.prob_deepfake:.4f}</b> ({prob_pct}%)<br><br>"
    )

    self.status_bar.showMessage("Готово.")
