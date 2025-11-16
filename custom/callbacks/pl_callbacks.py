from lightning.pytorch.callbacks import EarlyStopping, Callback, ModelCheckpoint

__all__ = ["CustomEarlyStopping",
           "CustomModelCheckpoint",
           "LearningRateLogger",
           "ResetLearningRateCallback"]


class CustomEarlyStopping(EarlyStopping):
    def __init__(self, start_epoch=0, **kwargs):
        super().__init__(**kwargs)
        self.start_epoch = start_epoch

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch >= self.start_epoch:
            super().on_validation_end(trainer, pl_module)


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, start_epoch=0, **kwargs):
        super().__init__(**kwargs)
        self.start_epoch = start_epoch

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch >= self.start_epoch:
            super().on_validation_end(trainer, pl_module)


class LearningRateLogger(Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """
        在每个训练批次开始时记录学习率
        """
        for i, optimizer in enumerate(trainer.optimizers):
            lr = optimizer.param_groups[0]['lr']
            trainer.logger.log_metrics({f"learning_rate_{i}": lr}, step=trainer.global_step)


class ResetLearningRateCallback(Callback):
    def __init__(self, reset_epoch=100, new_lr=1e-4):
        """
        初始化回调函数
        :param reset_epoch: 重置学习率的 epoch
        :param new_lr: 重置后的学习率
        """
        self.reset_epoch = reset_epoch
        self.new_lr = new_lr

    def on_train_epoch_start(self, trainer, pl_module):
        """
        在每个 epoch 开始时调用
        """
        # 检查是否到达指定的 epoch
        if trainer.current_epoch == self.reset_epoch:
            # 获取优化器
            optimizer = trainer.optimizers[0]
            # 重置学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.new_lr
