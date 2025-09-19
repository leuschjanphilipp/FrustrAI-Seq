from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

def get_callbacks(config: dict):
    early_stop = EarlyStopping(
        monitor=config["monitored_metric"],
        patience=config['early_stop_patience'],
        mode='min',
        verbose=True
    )
    checkpoint = ModelCheckpoint(
        monitor=config["monitored_metric"],
        dirpath=config['path_study'],
        filename=f"trial_{config["trial_number"]}_model",
        save_top_k=1,
        mode='min',
        save_weights_only=True
    )
    return [early_stop, checkpoint]