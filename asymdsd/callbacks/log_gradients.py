import lightning as L
from lightning.pytorch.loggers import WandbLogger


class LogGradients(L.Callback):
    def __init__(
        self,
        log_freq: int = 1000,
        log_graph: bool = False,
    ) -> None:
        self.log_freq = log_freq
        self.log_graph = log_graph

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        loggers = trainer.loggers

        for logger in loggers:
            if isinstance(logger, WandbLogger):
                logger.watch(
                    pl_module,
                    log="gradients",
                    log_freq=self.log_freq,
                    log_graph=self.log_graph,
                )
                break
