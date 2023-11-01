import torch
import torchvision
import torchvision.transforms as transforms
import os
import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional import accuracy
from torchvision import transforms
# Note - you must have torchvision installed for this example
from torchvision.datasets import CIFAR10, MNIST

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64

class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.dims = (3, 32, 32)
        self.num_classes = 10
        BATCH_SIZE = 4

        classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)


    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.cifar_test = CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=BATCH_SIZE)

    def test_dataloader(self):
        self.logger = wandb.Logger()
        return DataLoader(self.cifar_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
class Net(L.LightningModule):
    def __init__(self, channels, width, height, num_classes, hidden_size=64, learning_rate=2e-4):
        super().__init__()

        # We take in input dimensions as parameters and use those to dynamically build model.
        self.channels = channels
        self.width = width
        self.height = height
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        self.model = nn.Sequential(
            nn.Flatten(),
                nn.Linear(channels * width * height, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, num_classes),
            )

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
            #
            #x = self.model(x)
            #return F.log_softmax(x, dim=1)

    def training_step(self, batch):
            x, y = batch
            logits = self(x)
            loss = F.cross_entropy(logits, y)
            return loss

    def validation_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = F.nll_loss(logits, y)
            preds = torch.argmax(logits, dim=1)
            acc = accuracy(preds, y, task="multiclass", num_classes=10)

            self.logger.log_metrics({"val_loss": loss, "val_acc": acc}, step=self.global_step)
            return loss, acc

    '''def on_train_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        wandb.log({"avg_train_loss": avg_loss})

        def on_validation_epoch_end(self, outputs):
            avg_loss = torch.stack([x[0] for x in outputs]).mean()
            avg_acc = torch.stack([x[1] for x in outputs]).mean()
            wandb.log({"avg_val_loss": avg_loss, "avg_val_acc": avg_acc})'''

    def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            return optimizer    
    

from lightning.pytorch.callbacks import callback
dm = CIFAR10DataModule()
model = Net(*dm.dims, dm.num_classes, hidden_size=256)

#early stopping - EarlyStop
#lr callback
#model checkpoint = checkpoint callback - условия сохранения модели (раз в эпоху/лучшую на момент обучения (раз в эпоху))


trainer = L.Trainer(
    max_epochs=5,
    accelerator="auto",
    devices=1,
    logger=WandbLogger(),
)

trainer.fit(model, dm)
#wandb.run.finish()
wandb.finish()