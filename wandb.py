import wandb
from pytorch_lightning.loggers import WandbLogger

wandb.init(project="usermaryz homework")

trainer = L.Trainer(
    max_epochs=5,
    accelerator="auto",
    devices=1,
    logger=WandbLogger(),
)

trainer.fit(model, dm)
#wandb.run.finish()
wandb.finish()