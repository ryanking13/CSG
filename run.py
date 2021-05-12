import argparse
import datetime
import os
import subprocess as sp

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from data.dataloader import VISDA17DataModule
from model.siamese_net import SiameseNet
from model.resnet import resnet101


def parse_args():
    try:
        gpus_in_machine = len(
            sp.check_output(["nvidia-smi", "-L"]).strip().split(b"\n")
        )
    except:
        gpus_in_machine = 1

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output",
        default=os.environ.get("LOGS", "logs"),
        help="Output directory where checkpoint and log will be saved (default: %(default)s)",
    )
    parser.add_argument(
        "-r",
        "--root",
        default=os.path.join(os.environ.get("DATASETS", "datasets"), "visda17"),
        help="Root directory of the VisDA17 dataset (default: %(default)s)",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=30,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-bs",
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: %(default)s)",
    )
    parser.add_argument(
        "-wd",
        "--weight-decay",
        type=float,
        default=5e-4,
        help="Weight decay (default: %(default)s)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Optimizer momentum (default: %(default)s)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=12,
        help="Number of classes (default: %(default)s)",
    )
    parser.add_argument(
        "--emb-dim",
        type=int,
        default=128,
        help="Feature dimension (default: %(default)s)",
    )
    parser.add_argument(
        "--single-network",
        dest="siamese",
        action="store_false",
        default=True,
        help="Train with single network, for comparison",
    )
    parser.add_argument(
        "--nce-weight",
        type=float,
        default=0.1,
        help="Weight of nce loss (default: %(default)s)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        default=False,
        help="Do not train, evaluate model",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=gpus_in_machine,
        help="Number of gpus to use (default: All GPUs in the machine)",
    )
    parser.add_argument(
        "--resume", default=None, help="Resume from the given checkpoint"
    )
    parser.add_argument(
        "--dev-run",
        action="store_true",
        default=False,
        help="Run small steps to test whether model is valid",
    )
    parser.add_argument(
        "--exp-name",
        default="CSG-lightning",
        help="Experiment name used for a log directory name",
    )
    parser.add_argument(
        "--augmentation",
        default="rand_augment",
        help="Augmentations to use (default: %(default)s)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed (default: %(default)s)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("Args: ", args)

    pl.seed_everything(args.seed)

    encoder = resnet101

    model = SiameseNet(
        base_encoder=encoder,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        siamese=args.siamese,
        labmda_nce=args.nce_weight,
        stages=(3, 4),
        emb_dim=args.emb_dim,
        num_negatives=2 ** 16,
        encoder_momentum=0.999,
        softmax_temperature=0.07,
    )

    datamodule = VISDA17DataModule(
        data_dir=args.root,
        batch_size=args.batch_size,
        transforms=args.augmentation,
    )

    exp_name = args.exp_name

    # loggers, callbacks
    start_time = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
    logger = TensorBoardLogger(
        save_dir=args.output,
        name="tensorboard",
        version=exp_name + start_time,
        default_hp_metric=False,
    )
    checkpoint_monitor = ModelCheckpoint(
        monitor="val_acc1",
        mode="max",
        dirpath=os.path.join(args.output, "tensorboard", exp_name + start_time),
        filename=exp_name + "_{epoch:02d}_{val_loss:.2f}_{val_acc1:.2f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        gpus=args.num_gpus,
        max_epochs=args.epochs,
        check_val_every_n_epoch=1,
        logger=logger,
        callbacks=[
            lr_monitor,
            checkpoint_monitor,
        ],
        resume_from_checkpoint=args.resume,
        fast_dev_run=args.dev_run,
    )

    if args.eval_only:
        if args.resume is None:
            print("You must specify --resume <checkpoint>")
            exit(1)

        model = SiameseNet.load_from_checkpoint(args.resume)
        datamodule.setup(stage="val")
        trainer.validate(model, val_dataloaders=datamodule.val_dataloader())
    else:
        trainer.fit(
            model,
            datamodule,
        )


if __name__ == "__main__":
    main()
