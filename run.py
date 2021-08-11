import argparse
import datetime
import os
import subprocess as sp
import random

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

from data.dataloader import VISDA17DataModule, GTA5toCityscapesDataModule
from model.siamese_net import SiameseNet, SiameseNetSegmentation
from model.resnet import resnet101
from model.deeplab import deeplab101, deeplab50
import loss as ssl_loss


def parse_args():
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
        default=os.path.join(os.environ.get("DATASETS", "datasets")),
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
        "--task",
        default="classification",
        choices=["classification", "segmentation"],
        help="Task to run (default: %(default)s)",
    )
    parser.add_argument(
        "--encoder",
        default="resnet101",
        choices=["resnet101", "deeplab50", "deeplab101"],
        help="Backbone network to use (default: %(default)s)",
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
        "--eval-only",
        action="store_true",
        default=False,
        help="Do not train, evaluate model",
    )
    parser.add_argument(
        "--gpus",
        default=-1,
        help="GPUs to use (default: All GPUs in the machine)",
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
        "--seed", default="random", help="Random seed (default: %(default)s)"
    )
    parser.add_argument(
        "--fc-dim",
        type=int,
        default=512,
        help="Dimension of FC hidden layer (default: %(default)s)",
    )
    parser.add_argument(
        "--no-apool",
        dest="apool",
        action="store_const",
        default=True,
        const=False,
        help="Disable attentional pooling",
    )

    parser.add_argument(
        "--single-network",
        dest="siamese",
        action="store_false",
        default=True,
        help="Train with single network, for comparison",
    )
    parser.add_argument(
        "--stages",
        default=(3, 4),
        nargs="+",
        help="Feature stages to use in contrastive loss calculation",
    )
    parser.add_argument(
        "--emb-dim",
        type=int,
        default=128,
        help="Feature dimension (default: %(default)s)",
    )
    parser.add_argument(
        "--emb-depth",
        type=int,
        default=1,
        help="Depth of project layers (default: %(default)s)",
    )
    parser.add_argument(
        "--num-patches",
        type=int,
        default=1,
        help="Crop feature maps to NxN patches, for dense contrastive loss caculation (default: %(default)s)",
    )

    parser.add_argument(
        "--moco-weight",
        type=float,
        default=0.1,
        help="Weight of MoCo loss (default: %(default)s)",
    )
    parser.add_argument(
        "--moco-queue-size",
        type=int,
        default=2 ** 16,
        help="Queue size of MoCo (default: %(default)s)",
    )
    parser.add_argument(
        "--moco-momentum",
        type=float,
        default=0.999,
        help="Encoder momentum of MoCo (default: %(default)s)",
    )
    parser.add_argument(
        "--moco-temperature",
        type=float,
        default=0.07,
        help="Temperature of MoCo (default: %(default)s)",
    )

    return parser.parse_args()


def build_loss(hparams):
    return ssl_loss.MoCoLoss(
        embedding_dim=hparams.emb_dim,
        queue_size=hparams.moco_queue_size,
        momentum=hparams.moco_momentum,
        temperature=hparams.moco_temperature,
        scale_loss=hparams.moco_weight,
        queue_ids=hparams.stages,
    )


def main():
    args = parse_args()
    print("Args: ", args)

    seed = random.randint(0, 1e7) if args.seed == "random" else int(args.seed)
    pl.seed_everything(seed)

    encoder = {
        "resnet101": resnet101,
        "deeplab50": deeplab50,
        "deeplab101": deeplab101,
    }[args.encoder]
    task = args.task.lower()

    if task == "classification":
        _model = SiameseNet
        _datamodule = VISDA17DataModule
        monitor = "val_acc1"
    elif task == "segmentation":
        _model = SiameseNetSegmentation
        _datamodule = GTA5toCityscapesDataModule
        monitor = "val_iou"
    else:
        raise KeyError(f"Unknown task: {task}")

    model = _model(
        base_encoder=encoder,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        siamese=args.siamese,
        stages=args.stages,
        emb_dim=args.emb_dim,
        emb_depth=args.emb_depth,
        fc_dim=args.fc_dim,
        apool=args.apool,
        num_patches=args.num_patches,
        contrastive_loss=build_loss(args),
    )

    datamodule = _datamodule(
        root_dir=args.root,
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
        monitor=monitor,
        mode="max",
        dirpath=os.path.join(args.output, "tensorboard", exp_name + start_time),
        filename=exp_name + "_{epoch:02d}_{val_loss:.2f}_{val_acc1:.2f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=args.epochs,
        check_val_every_n_epoch=1,
        accelerator="ddp",
        logger=logger,
        callbacks=[
            lr_monitor,
            checkpoint_monitor,
        ],
        resume_from_checkpoint=args.resume,
        fast_dev_run=args.dev_run,
        plugins=[
            DDPPlugin(find_unused_parameters=True),
        ],
    )

    if args.eval_only:
        if args.resume is None:
            print("You must specify --resume <checkpoint>")
            exit(1)

        model = _model.load_from_checkpoint(
            args.resume, contrastive_loss=build_loss(args)
        )
        datamodule.setup(stage="val")
        trainer.validate(model, val_dataloaders=datamodule.val_dataloader())
    else:
        trainer.fit(
            model,
            datamodule,
        )


if __name__ == "__main__":
    main()
