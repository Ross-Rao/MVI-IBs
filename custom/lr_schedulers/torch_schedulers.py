from torch.optim.lr_scheduler import ReduceLROnPlateau

__all__ = [
    'CustomReduceLROnPlateau'
]


class CustomReduceLROnPlateau(ReduceLROnPlateau):
    """
    一个自定义的学习率调度器，功能与 ReduceLROnPlateau 相同，
    但可以在指定的 epoch 之后才开始监控并调整学习率。
    """

    def __init__(self, optimizer, start_epoch, **kwargs):
        super().__init__(optimizer, **kwargs)
        if not isinstance(start_epoch, int) or start_epoch < 0:
            raise ValueError(f"start_epoch 必须是一个非负整数，但接收到 {start_epoch}")
        self.start_epoch = start_epoch
        self._last_lr = [group['lr'] for group in optimizer.param_groups]  # 记录初始LR

    def step(self, metrics, epoch=None):
        if epoch is None:
            # 如果没有提供 epoch，我们尝试从 _step_count 获取当前的 epoch。
            # _step_count 在 ReduceLROnPlateau 内部每次 step 都会增加。
            current_epoch = self.last_epoch
        else:
            current_epoch = epoch

        if current_epoch < self.start_epoch:
            # 在达到 start_epoch 之前，不进行学习率调整，并保持内部状态同步
            # 如果不调用父类的 step 方法，ReduceLROnPlateau 的 _step_count 将不会增加
            # 从而导致在达到 start_epoch 时，_step_count 与实际 epoch 不匹配
            # 解决办法是让父类 step() 运行，但是让它的 LR 调整无效。
            # 我们可以临时修改当前 LR，并在父类 step() 执行后恢复。

            # 记录当前学习率
            original_lrs = [group['lr'] for group in self.optimizer.param_groups]

            # 临时将学习率设置为我们希望在启动前保持的值
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self._last_lr[i]

            # 调用父类的 step 方法，这将增加 _step_count
            super().step(metrics, epoch)

            # 恢复原始学习率
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = original_lrs[i]

            # 确保 _last_lr 始终反映当前学习率
            self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

        else:
            # 达到或超过 start_epoch，正常调用父类的 step 方法
            super().step(metrics, epoch)
            self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
