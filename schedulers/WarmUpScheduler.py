import torch


class WarmUpScheduler:
    """
    Basic warm up scheduler. Useful for transformers.
    """

    def __init__(self, optimizer, scheduler, start_lr=1e-5, goal_lr=1e-2, warmup_epochs=50):
        self.epoch = 0
        self.warmup_epochs = warmup_epochs
        self.scheduler = scheduler

        gamma = (goal_lr / start_lr) ** (1 / warmup_epochs)

        self.warmup_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=gamma, total_iters=warmup_epochs)

    def step(self):
        self.epoch += 1

        if self.epoch < self.warmup_epochs:
            self.warmup_scheduler.step()
        else:
            self.scheduler.step()
