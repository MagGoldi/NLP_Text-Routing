import logging

from transformers import TrainerCallback


class FileLoggingCallback(TrainerCallback):
    """Пересылает метрики/события HF Trainer в переданный logging.Logger (файл + консоль)."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def on_train_begin(self, args, state, control, **kwargs):
        self.logger.info("Training started: %s", args.output_dir)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.logger.info("step=%s %s", state.global_step, logs)

    def on_train_end(self, args, state, control, **kwargs):
        self.logger.info("Training finished after %s steps", state.global_step)
