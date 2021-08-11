import torch
from torchmetrics import Metric

# Code adapted from https://github.com/Shreeyak/pytorch-lightning-segmentation-template/blob/master/seg_lapa/metrics.py
class IoU(Metric):
    def __init__(self, num_classes: int = 11, normalize: bool = False):
        """Calculates the metrics iou, true positives and false positives/negatives for multi-class classification
        problems such as semantic segmentation.
        Because this is an expensive operation, we do not compute or sync the values per step.
        Forward accepts:
        - ``prediction`` (float or long tensor): ``(N, H, W)``
        - ``label`` (long tensor): ``(N, H, W)``
        Note:
            This metric produces a dataclass as output, so it can not be directly logged.
        """
        super().__init__(compute_on_step=False, dist_sync_on_step=False)

        self.num_classes = num_classes
        # Metric normally calculated on batch. If true, final metrics (tp, fn, etc) will reflect average values per image
        self.normalize = normalize

        self.acc_confusion_matrix = None  # The accumulated confusion matrix
        self.count_samples = None  # Number of samples seen
        # Use `add_state()` for attr to track their state and synchronize state across processes
        self.add_state(
            "acc_confusion_matrix",
            default=torch.zeros((self.num_classes, self.num_classes)),
            dist_reduce_fx="sum",
        )
        self.add_state("count_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, prediction: torch.Tensor, label: torch.Tensor):
        """Calculate the confusion matrix and accumulate it
        Args:
            prediction: Predictions of network (after argmax). Shape: [N, H, W]
            label: Ground truth. Each pixel has int value denoting class. Shape: [N, H, W]
        """
        assert prediction.shape == label.shape
        assert len(label.shape) == 3

        num_images = int(label.shape[0])

        label = label.view(-1).long()
        prediction = prediction.view(-1).long()

        # Calculate confusion matrix
        mask = (label >= 0) & (label < self.num_classes)
        conf_mat = torch.bincount(
            self.num_classes * label[mask] + prediction[mask],
            minlength=self.num_classes ** 2,
        )
        conf_mat = conf_mat.reshape((self.num_classes, self.num_classes))

        # Accumulate values
        self.acc_confusion_matrix += conf_mat
        self.count_samples += num_images

    def compute(self):
        """Compute the final IoU and other metrics across all samples seen"""
        # Normalize the accumulated confusion matrix, if needed
        conf_mat = self.acc_confusion_matrix
        if self.normalize:
            conf_mat = conf_mat / self.count_samples  # Get average per image

        # Calculate True Positive (TP), False Positive (FP), False Negative (FN) and True Negative (TN)
        tp = conf_mat.diagonal()
        fn = conf_mat.sum(dim=0) - tp
        fp = conf_mat.sum(dim=1) - tp
        total_px = conf_mat.sum()
        tn = total_px - (tp + fn + fp)

        # Calculate Intersection over Union (IoU)
        eps = 1e-6
        iou_per_class = (tp + eps) / (
            fn + fp + tp + eps
        )  # Use epsilon to avoid zero division errors
        iou_per_class[torch.isnan(iou_per_class)] = 0
        mean_iou = iou_per_class.mean()

        return mean_iou
