"""Code adapted from PytorchLightning/Lightnig-bolts and facebookresearch/vissl"""
import torch
from torch import nn
from torch.random import initial_seed


class MoCoLoss(nn.Module):
    def __init__(
        self,
        embedding_dim,
        queue_size,
        momentum,
        temperature,
        scale_loss,
        queue_ids,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        self.scale_loss = scale_loss
        self.queue_ids = queue_ids
        self.ddp = True
        self.initialized = False

        for queue_id in queue_ids:
            self.register_buffer(
                f"queue_{queue_id}",
                torch.randn(self.embedding_dim, self.queue_size),
            )
            self.register_buffer(
                f"queue_ptr_{queue_id}", torch.zeros(1, dtype=torch.long)
            )
            setattr(
                self,
                f"queue_{queue_id}",
                nn.functional.normalize(getattr(self, f"queue_{queue_id}"), dim=0),
            )

        self.criterion = nn.CrossEntropyLoss()

    def on_forward(self, *args, **kwargs):
        encoder_q = kwargs["encoder_q"]
        encoder_k = kwargs["encoder_k"]
        self._momentum_update_key_encoder(encoder_q, encoder_k)

    @torch.no_grad()
    def _momentum_update_key_encoder(self, encoder_q, encoder_k):
        """
        Momentum update of the key encoder
        """
        # only update mlp, freeze feature extractor
        for param_q, param_k in zip(
            encoder_q.mlp.parameters(), encoder_k.mlp.parameters()
        ):
            em = self.momentum
            param_k.data = param_k.data * em + param_q.data * (1.0 - em)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, stage):
        # gather keys before updating queue
        if self.ddp:
            keys = concat_all_gather(keys)

        queue = getattr(self, f"queue_{stage}")
        queue_ptr = getattr(self, f"queue_ptr_{stage}")

        self._dequeue_and_euqueue_stage(keys, queue, queue_ptr)

    @torch.no_grad()
    def _dequeue_and_euqueue_stage(self, keys, queue, queue_ptr):
        batch_size = keys.shape[0]

        ptr = int(queue_ptr)
        # assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        queue_ptr[0] = ptr

    def forward(self, q, k, *args, **kwargs):
        if not self.initialized:
            for queue_id in self.queue_ids:
                setattr(
                    self,
                    f"queue_{queue_id}",
                    getattr(self, f"queue_{queue_id}").to(q.device),
                )
            self.initialized = True

        stage = kwargs["stage"]

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
        logits /= self.temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        labels = labels.type_as(logits)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, stage)

        return self.scale_loss * self.criterion(logits.float(), labels.long())


# TODO: Remove duplicate
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
