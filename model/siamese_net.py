"""Adapted from MoCo implementation of PytorchLightning/lightning-bolts"""
``
from typing import Union, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pl_bolts.metrics import mean, precision_at_k
from torchmetrics import IoU

from .resnet import BasicBlock, Bottleneck
from .deeplab import ASPP

class SiameseNet(pl.LightningModule):
    def __init__(
        self,
        base_encoder: Union[str, torch.nn.Module] = "resnet101",
        num_classes: int = 12,
        emb_dim: int = 128,
        emb_depth: int = 1,
        fc_dim: int = 512,
        num_patches: int = 1,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        batch_size: int = 32,
        stages: Union[List, Tuple] = (3, 4),
        siamese: bool = True,
        flip_on_validation: bool = False,
        apool: bool = True,
        contrastive_loss = None,
        *args,
        **kwargs,
    ):
        """
        Args:
            base_encoder: base network which consists siamese network
            num_classes: # of classes
            emb_dim: projector dimension which will be used for calculating contrastive loss
            emb_depth: projector depth
            learning_rate: the learning rate
            momentum: optimizer momentum
            weight_decay: optimizer weight decay
            batch_size: batch size
            stages: base encoder (resnet) stages to calculate contrastive loss
            siamese: whether or not to use contrastive loss
            flip_on_validation: on validation, use the mean classification result of two flipped image
            apool: use attentional pooling instead of GAP
            contrastive_loss: contrastive loss function to use
        """
        super().__init__()
        self.save_hyperparameters()

        assert (
            not self.hparams.siamese or self.hparams.contrastive_loss is not None
        ), f"Contrastive loss must be specified when using siamese network"

        self.contrastive_loss = contrastive_loss(ddp=True) if contrastive_loss is not None else None

        # create encoders
        self.encoder_q, self.encoder_k = self._init_encoders(self.hparams.base_encoder, self.hparams.num_classes)

        self._override_classifier()
        self._attach_projector()
    
    def _override_classifier(self):
        # hack: brute-force replacement

        # Note: this method assumes that base encoder IS Pytorch ResNet (OR it has 'fc' layer)
        #       if you want to use other encoder, you must override this method
        classifier_layer = self.encoder_q.fc
        dim_fc = classifier_layer.weight.shape[1]
        dim_head = self.hparams.fc_dim
        
        self.encoder_q.fc = nn.Sequential(
            nn.Linear(dim_fc, dim_head),
            nn.ReLU(),
            nn.Linear(dim_head, self.hparams.num_classes),
        )
        self.encoder_k.fc = nn.Sequential(
            nn.Linear(dim_fc, dim_head),
            nn.ReLU(),
            nn.Linear(dim_head, self.hparams.num_classes),
        )
    
    def _attach_projector(self):
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

            emb_q = []
            emb_k = []
            for _ in range(self.hparams.emb_depth):
                emb_q.append(nn.Linear(dim_mlp, dim_mlp))
                emb_q.append(nn.ReLU())
                emb_k.append(nn.Linear(dim_mlp, dim_mlp))
                emb_k.append(nn.ReLU())
            
            emb_q.append(nn.Lienar(dim_mlp, self.hparams.emb_dim))
            emb_k.append(nn.Lienar(dim_mlp, self.hparams.emb_dim))

            mlp_q[f"mlp_{stage}"] = nn.Sequential(
                *emb_q,
            )
            mlp_k[f"mlp_{stage}"] = nn.Sequential(
                *emb_k,
            )

        self.encoder_q.mlp = nn.ModuleDict(mlp_q)
        self.encoder_k.mlp = nn.ModuleDict(mlp_k)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def _init_encoders(self, base_encoder, num_classes):
        encoder_q = base_encoder(pretrained=True, num_classes=num_classes)
        encoder_k = base_encoder(pretrained=True, num_classes=num_classes)

        return encoder_q, encoder_k

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
        self.contrastive_loss.on_forward(
            encoder_q=self.encoder_q,
            encoder_k=self.encoder_k,
        )

        features = []

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            # When chunking is enabled, shuffle doesn't work
            # if self.trainer.use_ddp or self.trainer.use_ddp2:
            #     img_k, idx_unshuffle = self._batch_shuffle_ddp(img_k)

            _, features_k = self.encoder_k(img_k)
            if self.hparams.apool:
                _, features_k_on_q = self.encoder_q(img_k)

        for idx, stage in enumerate(self.hparams.stages):
            features_q[stage] = self.chunk_feature(features_q[stage], self.hparams.num_patches) 
            if self.hparams.apool:
                q = self.adaptive_pool(features_q[stage], features_q[stage])
            else:
                q = self.avgpool(features_q[stage])

            q = torch.flatten(q, 1)
            q = self.encoder_q.mlp[f"mlp_{stage}"](q)
            q = nn.functional.normalize(q, dim=1)

            with torch.no_grad():
                features_k[stage] = self.chunk_feature(features_k[stage], self.hparams.num_patches)
                features_k_on_q[stage] = self.chunk_feature(features_k_on_q[stage], self.hparams.num_patches)
                if self.hparams.apool:
                    k = self.adaptive_pool(features_k[stage], features_k_on_q[stage])
                else:
                    k = self.avgpool(features_k[stage])

                # undo shuffle
                # When chunking is enabled, shuffle doesn't work
                # if self.trainer.use_ddp or self.trainer.use_ddp2:
                #     k = self._batch_unshuffle_ddp(k, idx_unshuffle)

                k = torch.flatten(k, 1)
                k = self.encoder_k.mlp[f"mlp_{stage}"](k)
                k = nn.functional.normalize(k, dim=1)

            features.append({
                "q": q,
                "k": k,
                "stage": stage,
            })

        return y_pred, features

    def training_step(self, batch, batch_idx):
        # Freeze batchnorm layers
        self.apply(set_bn_eval)

        (img_1, img_2), labels = batch

        y_pred, features = self(img_q=img_1, img_k=img_2)

        loss_class = F.cross_entropy(y_pred, labels.long())
        loss_contrastive = 0.0

        if self.hparams.siamese:
            loss_contrastive = sum([
                self.contrastive_loss(f["q"], f["k"], stage=f["stage"] for f in features
            ])

        acc1, acc5 = precision_at_k(y_pred, labels, top_k=(1, 5))

        loss = loss_class + loss_contrastive
        log = {
            "train_loss": loss,
            "train_loss_class": loss_class,
            "train_loss_contrastive", loss_contrastive,
            "train_acc1": acc1,
            "train_acc5": acc5
        }
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
                f"[Epoch {self.current_epoch}]: [Val loss: {val_loss:.3f} / Val acc1: {val_acc1:.3f} / Val acc5: {val_acc5:.3f}]"
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
    
    def chunk_feature(self, feature, num_chunks):
        if num_chunks == 1:
            return feature
        
        chunks = torch.chunk(feature, num_chunks, dim=2)
        chunks = [torch.chunk(c, num_chunks, dim=3) for c in chunks]

        chunked_feature = []
        for c in chunks:
            chunked_feature += c
        
        return torch.cat(chunked_feature, dim=0)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer


class SiameseNetSegmentation(SiameseNet):
    def __init__(
        self,
        base_encoder,
        num_classes=12,
        emb_dim=128,
        emb_depth=1,
        learning_rate=0.1,
        momentum=0.9,
        weight_decay=5e-4,
        batch_size=32,
        stages=(3, 4),
        flip_on_validation=False,
        fc_dim=512,
        num_patches=8,
        apool=True,
        contrastive_loss=None,
        *args,
        **kwargs,
    ):
        super().__init__(base_encoder, num_classes, emb_dim, emb_depth, learning_rate, momentum,
                        weight_decay, batch_size, stages, flip_on_validation, fc_dim, num_patches, apool, contrastive_loss,
                        *args, **kwargs)
        
        self.iou = IoU(num_classes=self.hparams.num_classes, ignore_index=255)
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction="mean")
    
    def _override_classifier(self):
        self.encoder_q.aspp = ASPP(2048, self.hparams.num_classes, [6, 12, 18, 24])
        self.encoder_k.aspp = ASPP(2048, self.hparams.num_classes, [6, 12, 18, 24])
    
    def training_step(self, batch, batch_idx):
        self.apply(set_bn_eval)

        (img_1, img_2), labels = batch
        y_pred, features = self(img_q=img_1, img_k=img_2)

        loss_class = self.criterion(y_pred, labels.long())
        loss_contrastive = 0.0

        if self.hparams.siamese:
            loss_contrastive = sum([
                self.contrastive_loss(f["q"], f["k"], stage=f["stage"] for f in features
            ])

        loss = loss_class + loss_contrastive
        log = {
            "train_loss": loss,
            "train_loss_class": loss_class,
            "train_loss_contrastive", loss_contrastive,
            "train_acc1": acc1,
            "train_acc5": acc5
        }
        self.log_dict(log, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        img, labels = batch
        y_pred, _ = self.encoder_q(img)

        loss = self.criterion(y_pred, labels.long())

        self.iou(y_pred.argmax(dim=1), labels)

        results = {"val_loss": loss}
        return results

    def validation_epoch_end(self, results):

        val_loss = mean(results, "val_loss")

        val_iou = self.iou.compute()
        self.iou.reset()

        log = {"val_loss": val_loss, "val_iou": val_iou}
        self.log_dict(log, sync_dist=True)
        try:
            # LightningModule.print() has an issue on printing when validation mode
            self.print(
                f"[Epoch {self.current_epoch}]: [Val loss: {val_loss:.3f} / Val mIoU: {val_iou:.3f}]"
            )
        except:
            pass

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


def set_bn_eval(module):
    if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
        module.eval()