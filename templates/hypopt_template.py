import optuna

from utils.train_utils import train_one_epoch, evaluate


def objective(trial: optuna.Trial):
    print(f"Trial #{trial.number} : {trial.params}")

    trn_dataloader = None
    val_dataloader = None

    #### Define hyperparamters here
    epochs = trial.suggest_int("epochs", 10, 1000)
    ####

    net = None
    criterion = None
    optimizer = None
    scheduler = None
    metric = "accuracy"

    for epoch in range(epochs):
        train_one_epoch(net, trn_dataloader, criterion, optimizer, scheduler=scheduler, grad_clip=None)
        val = evaluate(net, val_dataloader, metrics=[metric])

        trial.report(val[metric], epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

    val = evaluate(net, val_dataloader, metrics=[metric])
    return val[metric]


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=100)
