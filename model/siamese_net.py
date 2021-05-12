"""Adapted from MoCo implementation of PytorchLightning/lightning-bolts"""

from typing import Union, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pl_bolts.metrics import mean, precision_at_k

from .resnet import BasicBlock, Bottleneck


class SiameseNet(pl.LightningModule):
    def __init__(
        self,
        base_encoder: Union[str, torch.nn.Module] = "resnet101",
        emb_dim: int = 128,
        num_negatives: int = 65536,
        encoder_momentum: float = 0.999,
        softmax_temperature: float = 0.07,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        batch_size: int = 32,
        stages: Union[List, Tuple] = (3, 4),
        lambda_nce: float = 0.1,
        siamese: bool = True,
        flip_on_validation: bool = True,
        apool: bool = True,
        *args,
        **kwargs,
    ):
        """
        Args:
            base_encoder: base network which consists siamese network
            emb_dim: feature dimension which will be used for calculating NCE-loss
            num_negatives: queue size; number of negative keys
            encoder_momentum: moco momentum of updating key encoder
            softmax_temperature: softmax temperature for NCE-loss
            learning_rate: the learning rate
            momentum: optimizer momentum
            weight_decay: optimizer weight decay
            batch_size: batch size
            stages: base encoder (resnet) stages to calculate NCE-loss
            lambda_nce: weight of NCE-loss
            use_nce: use NCE-loss for training, if false, it is equivalent to training a single netowork
            flip_on_validation: on validation, use the mean classification result of two flipped image
            apool: use attentional pooling instead of GAP
        """
        super().__init__()
        self.save_hyperparameters()

        self.encoder_q, self.encoder_k = self._init_encoders(self.hparams.base_encoder)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # hack: brute-force replacement
        dim_fc = self.encoder_q.fc.weight.shape[1]
        self.encoder_q.fc = nn.Sequential(
            nn.Linear(dim_fc, 512),
            nn.ReLU(),
            nn.Linear(512, self.hparams.num_classes),
        )
        self.encoder_k.fc = nn.Sequential(
            nn.Linear(dim_fc, 512),
            nn.ReLU(),
            nn.Linear(512, self.hparams.num_classes),
        )

        # add mlp layer for contrastive loss
        mlp_q = {}
        mlp_k = {}
        for stage in self.hparams.stages:
            block = getattr(self.encoder_q, f"layer{stage}")[-1]

            if isinstance(block, Bottleneck):
                dim_mlp = block.conv3.weight.shape[0]
            elif isinstance(block, BasicBlock):
                dim_mlp = block.conv2.weight.shape[0]
            else:
                raise NotImplementedError(f"{type(block)} not supported.")

            mlp_q[f"mlp_{stage}"] = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp),
                nn.ReLU(),
                nn.Linear(dim_mlp, self.hparams.emb_dim),
            )
            mlp_k[f"mlp_{stage}"] = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp),
                nn.ReLU(),
                nn.Linear(dim_mlp, self.hparams.emb_dim),
            )

        self.encoder_q.mlp = nn.ModuleDict(mlp_q)
        self.encoder_k.mlp = nn.ModuleDict(mlp_k)

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        for stage in self.hparams.stages:
            self.register_buffer(
                f"queue_{stage}",
                torch.randn(self.hparams.emb_dim, self.hparams.num_negatives),
            )
            self.register_buffer(f"queue_ptr_{stage}", torch.zeros(1, dtype=torch.long))
            setattr(
                self,
                f"queue_{stage}",
                nn.functional.normalize(getattr(self, f"queue_{stage}"), dim=0),
            )

    def _init_encoders(self, base_encoder):
        encoder_q = base_encoder(pretrained=True)
        encoder_k = base_encoder(pretrained=True)

        return encoder_q, encoder_k

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        # only update mlp, freeze feature extractor
        for param_q, param_k in zip(
            self.encoder_q.mlp.parameters(), self.encoder_k.mlp.parameters()
        ):
            em = self.hparams.encoder_momentum
            param_k.data = param_k.data * em + param_q.data * (1.0 - em)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, stage):
        # gather keys before updating queue
        if self.trainer.use_ddp or self.trainer.use_ddp2:
            keys = concat_all_gather(keys)

        queue = getattr(self, f"queue_{stage}")
        queue_ptr = getattr(self, f"queue_ptr_{stage}")

        self._dequeue_and_euqueue_stage(keys, queue, queue_ptr)

    @torch.no_grad()
    def _dequeue_and_euqueue_stage(self, keys, queue, queue_ptr):
        batch_size = keys.shape[0]

        ptr = int(queue_ptr)
        assert self.hparams.num_negatives % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.hparams.num_negatives  # move pointer

        queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):  # pragma: no cover
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):  # pragma: no cover
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, img_q, img_k):
        y_pred, features_q = self.encoder_q(img_q)
        self._momentum_update_key_encoder()  # update the key encoder

        logits_all = []
        labels_all = []

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            if self.trainer.use_ddp or self.trainer.use_ddp2:
                img_k, idx_unshuffle = self._batch_shuffle_ddp(img_k)

            _, features_k = self.encoder_k(img_k)
            if self.hparams.apool:
                _, features_k_on_q = self.encoder_q(img_k)

        for idx, stage in enumerate(self.hparams.stages):
            if self.hparams.apool:
                q = self.adaptive_pool(features_q[stage], features_q[stage])
            else:
                q = self.avgpool(features_q[stage])

            q = torch.flatten(q, 1)
            q = self.encoder_q.mlp[f"mlp_{stage}"](q)
            q = nn.functional.normalize(q, dim=1)

            with torch.no_grad():
                if self.hparams.apool:
                    k = self.adaptive_pool(features_k[stage], features_k_on_q[stage])
                else:
                    k = self.avgpool(features_k[stage])

                # undo shuffle
                if self.trainer.use_ddp or self.trainer.use_ddp2:
                    k = self._batch_unshuffle_ddp(k, idx_unshuffle)

                k = torch.flatten(k, 1)
                k = self.encoder_k.mlp[f"mlp_{stage}"](k)
                k = nn.functional.normalize(k, dim=1)

            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum(
                "nc,ck->nk", [q, getattr(self, f"queue_{stage}").clone().detach()]
            )

            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.hparams.softmax_temperature

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long)
            labels = labels.type_as(logits)

            logits_all.append(logits)
            labels_all.append(labels)

            # dequeue and enqueue
            self._dequeue_and_enqueue(k, stage)

        return y_pred, logits_all, labels_all

    def training_step(self, batch, batch_idx):
        # Note: setting eval mode in training step is important for performance,
        #       which freezes batchnorm weights to pretrained weights.
        #       Further analysis required.
        self.eval()

        (img_1, img_2), labels = batch

        y_pred, output, target = self(img_q=img_1, img_k=img_2)
        loss_class = F.cross_entropy(y_pred, labels.long())
        loss_nce = sum(
            [F.cross_entropy(o.float(), t.long()) for o, t in zip(output, target)]
        )

        loss = loss_class
        if self.hparams.siamese:
            loss += self.hparams.lambda_nce * loss_nce

        acc1, acc5 = precision_at_k(y_pred, labels, top_k=(1, 5))

        log = {"train_loss": loss, "train_acc1": acc1, "train_acc5": acc5}
        self.log_dict(log, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, labels = batch
        y_pred, _ = self.encoder_q(img)

        if self.hparams.flip_on_validation:
            y_pred_flip, _ = self.encoder_q(torch.flip(img, dims=(3,)))
            y_pred = (y_pred + y_pred_flip) / 2

        loss = F.cross_entropy(y_pred, labels.long())

        acc1, acc5 = precision_at_k(y_pred, labels, top_k=(1, 5))

        results = {"val_loss": loss, "val_acc1": acc1, "val_acc5": acc5}
        return results

    def validation_epoch_end(self, results):
        val_loss = mean(results, "val_loss")
        val_acc1 = mean(results, "val_acc1")
        val_acc5 = mean(results, "val_acc5")

        log = {"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5}
        self.log_dict(log, sync_dist=True)
        try:
            # LightningModule.print() has an issue on printing when validation mode
            self.print(
                f"[Epoch {self.current_epoch}]: [Val loss: {val_loss} / Val acc1: {val_acc1} / Val acc5: {val_acc5}]"
            )
        except:
            pass

    def adaptive_pool(self, features, attention_base):
        assert features.shape == attention_base.shape

        batch_size = features.shape[0]
        gap = torch.flatten(self.avgpool(attention_base), start_dim=1)
        attention = torch.einsum("nchw,nc->nhw", [attention_base, gap])
        attention /= torch.einsum("nhw->n", [attention]).view(batch_size, 1, 1)
        features_with_attention = torch.einsum("nchw,nhw->nchw", [features, attention])
        return torch.einsum("nchw->nc", [features_with_attention])

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
