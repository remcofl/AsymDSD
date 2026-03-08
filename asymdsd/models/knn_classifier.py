from typing import Any

import torch

from ..components import *
from ..data import PCFieldKey
from .base_embedding_classifier import BaseEmbeddingClassifier


class KNNClassifier(BaseEmbeddingClassifier):
    CLASSIFIER_NAME = "knn"

    def __init__(
        self, n_neighbors: int = 20, classifier_name=CLASSIFIER_NAME, **kwargs
    ):
        super().__init__(classifier_name=classifier_name, **kwargs)
        self.n_neighbors = n_neighbors
        self.hparams["n_neighbors"] = n_neighbors

    def validation_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        target_embeddings = self.extract_embeddings(batch)[0]
        val_labels: torch.Tensor = batch[PCFieldKey.CLOUD_LABEL]
        B = len(val_labels)

        similarity = target_embeddings @ self.embeddings.T  # type: ignore
        topk = similarity.topk(k=self.n_neighbors, dim=1, largest=True, sorted=True)

        indices = topk.indices
        topk_labels = self.labels[indices]

        counts: torch.Tensor = torch.zeros(
            (B, self.num_classes),  # type: ignore
            device=val_labels.device,
        )
        counts.scatter_(dim=1, index=topk_labels, value=1, reduce="add")

        # Update all topk metrics
        for metric in self.top_acc_metrics.values():
            metric.update(counts, val_labels)

        if self.log_mean_acc:
            self.mean_acc.update(counts, val_labels)

        return {
            "pred_indices": counts.argmax(dim=1),
            "target_indices": val_labels,
        }

    def predict_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        embeddings = self.extract_embeddings(batch)[0]
        B = embeddings.size(0)

        similarity = embeddings @ self.embeddings.T  # type: ignore
        topk = similarity.topk(k=self.n_neighbors, dim=1, largest=True, sorted=True)

        indices = topk.indices
        topk_labels = self.labels[indices]

        counts: torch.Tensor = torch.zeros(
            (B, self.num_classes),  # type: ignore
            device=embeddings.device,
        )
        counts.scatter_(dim=1, index=topk_labels, value=1, reduce="add")

        return {
            "pred_indices": counts.argmax(dim=1),
        }

    def fit_model(self):
        pass
