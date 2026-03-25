"""
Multi-category data loading utilities.

Creates a minimal datamodule-like object compatible with anomalib Engine.fit(),
by concatenating TRAIN splits from multiple MVTec categories.
"""

from pathlib import Path
from typing import List, Optional, Union

from torch.utils.data import ConcatDataset, DataLoader, Subset
from anomalib.data import MVTecAD

from utils.data_utils import download_mvtec_ad


class MultiCategoryMVTecAD:
    """
    Concatenates train datasets across multiple categories.
    Exposes:
      - setup()
      - train_dataloader()
      - val_dataloader() (optional)
    """

    def __init__(
        self,
        root: Union[str, Path],
        categories: List[str],
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        limit_train_images_per_category: Optional[int] = None,
        auto_download: bool = True,
        num_workers: int = 0,
    ):
        self.root = Path(root)
        self.categories = categories
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.limit_train_images_per_category = limit_train_images_per_category
        self.auto_download = auto_download
        self.num_workers = num_workers

        self._train_dataset = None
        self._val_dataset = None
        self._train_collate_fn = None
        self._val_collate_fn = None

    def setup(self, stage: Optional[str] = None):
        if self.auto_download:
            self.root = download_mvtec_ad(self.root)

        train_sets = []
        val_sets = []

        for cat in self.categories:
            dm = MVTecAD(
                root=str(self.root),
                category=cat,
                train_batch_size=self.train_batch_size,
                eval_batch_size=self.eval_batch_size,
            )
            dm.setup()

            train_loader = dm.train_dataloader()
            train_ds = train_loader.dataset

            if self._train_collate_fn is None:
                self._train_collate_fn = train_loader.collate_fn

            # optional val
            try:
                val_loader = dm.val_dataloader()
                val_ds = val_loader.dataset
                if self._val_collate_fn is None:
                    self._val_collate_fn = val_loader.collate_fn
            except Exception:
                val_ds = None

            if self.limit_train_images_per_category is not None:
                n = min(self.limit_train_images_per_category, len(train_ds))
                train_ds = Subset(train_ds, range(n))

            train_sets.append(train_ds)
            if val_ds is not None:
                val_sets.append(val_ds)

        if not train_sets:
            raise ValueError("No train datasets found for the provided categories.")

        self._train_dataset = ConcatDataset(train_sets)
        self._val_dataset = ConcatDataset(val_sets) if val_sets else None

    def train_dataloader(self) -> DataLoader:
        if self._train_dataset is None:
            raise RuntimeError("Call setup() before requesting train_dataloader().")

        return DataLoader(
            self._train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._train_collate_fn,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self._val_dataset is None:
            return None

        return DataLoader(
            self._val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._val_collate_fn,
        )