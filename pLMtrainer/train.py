from pytorch_lightning import Trainer

def train(trainer, model, datamodule):

    trainer = Trainer(gpus=-1,
                      distributed_backend='ddp',
                      max_epochs=5,
                      #logger=logger,
                      #callbacks=callbacks,
                      gradient_clip_val=1,
                      enable_progress_bar=True,
                      deterministic=True
                      )

    trainer.fit(model, datamodule=datamodule)

