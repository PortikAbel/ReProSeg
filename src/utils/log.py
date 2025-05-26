import logging
import traceback
import typing
import warnings
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter


class BasicLog(logging.Logger):
    def __init__(
        self, log_dir: Path, name: str, disable_console: bool = False
    ):  # Store log in log_dir
        super().__init__(name)
        self._log_dir = log_dir

        # Ensure the directories exist
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._log_file = self._log_dir / "log.txt"

        if self._log_file.is_file():
            # make log file empty if it already exists
            self._log_file.write_text("")

        log_formatter = logging.Formatter(
            "[%(asctime)s - %(levelname)-7s] %(message)s", "%Y-%m-%d %H:%M:%S"
        )

        file_handler = logging.FileHandler(self._log_file)
        file_handler.setFormatter(log_formatter)
        self.addHandler(file_handler)

        if not disable_console:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(log_formatter)
            self.addHandler(stream_handler)

        warnings.showwarning = lambda message, *_: self.warning(
            f"{type(message).__name__}: {message}"
        )

        self.info(f"Log dir: {self.log_dir}")

    @property
    def log_dir(self):
        return self._log_dir

    def _log(
        self,
        level: int,
        msg: typing.Any,
        args,
        exc_info=None,
        extra: typing.Mapping[str, object] = None,
        stack_info: bool = False,
        stacklevel: int = 1,
    ):
        """
        Write a message to the log file

        :param level: the level of the message, e.g. logging.INFO
        :param msg: the message string to be written to the log file
        :param args: arguments for the message
        :param exc_info: exception info. Defaults to None
        :param extra: extra information. Defaults to None
        :param stack_info: whether to include stack info. Defaults to False
        :param stacklevel: the stack level. Defaults to 1
        """
        if type(msg) is not str:
            msg = str(msg)
        indent = msg[: len(msg) - len(msg.lstrip())]
        indent = indent.replace("\n", "")
        msg = msg.strip()
        for line in msg.splitlines():
            super()._log(
                level,
                f"{indent}{line}",
                args,
                exc_info=exc_info,
                extra=extra,
                stack_info=stack_info,
                stacklevel=stacklevel,
            )

    def exception(
        self,
        msg: Exception,
        *args,
        exc_info=True,
        stack_info=False,
        stacklevel=1,
        extra=None,
    ):
        """
        Log an exception.

        :param msg:
        :param args:
        :param exc_info:
        :param stack_info:
        :param stacklevel:
        :param extra:
        :return:
        """
        self.error(f"{type(msg).__name__} {msg}")
        self.error(traceback.format_exc())


class Log(BasicLog):
    """
    Object for managing the log directory
    """

    def __init__(
        self, log_dir: Path, name: str, disable_console: bool = False
    ):  # Store log in log_dir
        super().__init__(log_dir, name, disable_console)
        self._logs = dict()

        # Ensure the directories exist
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self.prototypes_dir.mkdir(parents=True, exist_ok=True)

        self._tqdm_file = (self._log_dir / "tqdm.txt").open(mode="w")
        self._tensorboard_writer = SummaryWriter(log_dir=self.tensorboard_dir)

    @property
    def tqdm_file(self):
        return self._tqdm_file

    @property
    def checkpoint_dir(self):
        return self._log_dir / "checkpoints"

    @property
    def metadata_dir(self):
        return self._log_dir / "metadata"
    
    @property
    def tensorboard_dir(self):
        return self._log_dir / "tensorboard"
    
    @property
    def prototypes_dir(self):
        return self._log_dir / "prototypes"
    
    def tb_scalar(self, tag, value, step):
        self._tensorboard_writer.add_scalar(tag, value, step)

    def model_checkpoint(self, state_dict, name):
        torch.save(state_dict, self.checkpoint_dir / name)

    def close(self):
        self._tqdm_file.close()
        self._tensorboard_writer.close()
