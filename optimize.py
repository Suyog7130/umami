"""
Docstring for optimize
"""

import optuna

def objective(trial):
    # Suggest hyperparameters
    n_cnn = trial.suggest_int("n_cnn", 2, 4)
    n_fc = trial.suggest_int("n_fc", 1, 3)
    base_out = trial.suggest_categorical("base_out", [32, 64, 128])
    cond_dim = trial.suggest_categorical("cond_dim", [0, 16])
    # You can add more suggestions as needed

    # Build config dict
    config = {
        "code": "2C2E1D",
        "kwargs": {
            "input_shape": (1, 2048),
            "latent_dim": 128,
            "num_classes": 10,
            "activation": "silu",
            "use_batchnorm": True,
            "dropout_p": 0.1,
            "n_layers_cnn_encoder": n_cnn,
            "n_layers_cnn_decoder": n_cnn,
            "n_layers_pre_fc": n_fc,
            "n_layers_post_fc": max(1, n_fc - 1),
            "encoder_has_cnn": True,
            "decoder_has_cnn": True,
            "encoder_has_fc": True,
            "decoder_has_fc": True,
            "pre_fc_sizes": [2048 + 10, 512, 256][: n_fc + 1],
            "post_fc_sizes": [256 + 10, 128] if n_fc >= 1 else [2048 + 10, 128],
            "cnn_in_channels": [1] + [base_out, base_out * 2][: max(0, n_cnn - 1)],
            "cnn_out_channels": [base_out] + [base_out * 2][: max(0, n_cnn - 1)],
            "cnn_kernel_size": [7] + [5] * (n_cnn - 1),
            "cnn_dilation": [1] * n_cnn,
            "cnn_pool_kernel_size": [2] * n_cnn,
            "n_conditioners": 0 if cond_dim == 0 else 2,
            "cond_dim": cond_dim if cond_dim else None,
        },
        "encoder_overrides": {1: {"has_cnn": False}},
        "tag": f"optuna_cnn{n_cnn}_fc{n_fc}_cond{cond_dim}_base{base_out}"
    }

    # Build and train model for a few{)} epochs, return validation loss
    trainer = MultiTrainer(
        train_data=train_loader,
        val_data=val_loader,
        epochs_stage1=3,  # Short for speed
        epochs_stage2=0,  # No retrain
        prune_every=100,  # No pruning
        topk_save_per_epoch=0,
        amp=True,
        compile_model=False,
        auto_continue=False,
    )
    # Train and get val loss
    model = trainer._build_model(config)
    for epoch in range(3):
        trainer._train_or_eval_epoch(model, train_loader, train=True)
    val_loss = trainer._train_or_eval_epoch(model, val_loader, train=False)
    return val_loss

# Run Optuna study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

print("Best trial:", study.best_trial.params)
