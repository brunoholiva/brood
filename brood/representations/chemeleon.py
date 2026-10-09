"""CheMeleon learned molecular fingerprint transformer.

Wraps the pretrained chemprop BondMessagePassing MPNN from
JacksonBurns/chemeleon as an sklearn-compatible TransformerMixin.
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
from chemprop import featurizers, nn
from chemprop.data import BatchMolGraph
from chemprop.models import MPNN
from chemprop.nn import RegressionFFN
from rdkit.Chem import Mol, MolFromSmiles
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm


class CheMeleonFingerprint(BaseEstimator, TransformerMixin):
    """sklearn transformer that featurizes molecules with CheMeleon embeddings.

    The embedding dimension is determined by the pretrained model
    (``mp.output_dim``, typically 2048). Downloads the checkpoint on
    first use (~45 MB) and caches it under ``~/.chemprop/chemeleon_mp.pt``.

    Parameters
    ----------
    device : str | torch.device | None, optional
        Device for model inference. If None, uses the default device.
    batch_size : int, default=256
        Molecules per forward pass, so large sets do not build one giant
        graph in memory.
    """

    def __init__(
        self,
        device: str | torch.device | None = None,
        batch_size: int = 256,
    ):
        self.device = device
        self.batch_size = batch_size
        self._model: MPNN | None = None
        self._featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    def _load_model(self) -> MPNN:
        if self._model is not None:
            return self._model
        ckpt_dir = Path.home() / ".chemprop"
        ckpt_dir.mkdir(exist_ok=True)
        mp_path = ckpt_dir / "chemeleon_mp.pt"
        if not mp_path.exists():
            urlretrieve(
                "https://zenodo.org/records/15460715/files/chemeleon_mp.pt",
                mp_path,
            )
        chemeleon_mp = torch.load(mp_path, weights_only=True)
        mp = nn.BondMessagePassing(**chemeleon_mp["hyper_parameters"])
        mp.load_state_dict(chemeleon_mp["state_dict"])
        self._model = MPNN(
            message_passing=mp,
            agg=nn.MeanAggregation(),
            predictor=RegressionFFN(input_dim=mp.output_dim),
        )
        self._model.eval()
        if self.device is not None:
            self._model.to(device=self.device)
        return self._model

    def fit(self, X=None, y=None):
        """No-op: model is loaded lazily during transform."""
        return self

    def transform(self, X: list[str | Mol]) -> np.ndarray:
        """Featurize molecules into CheMeleon learned embeddings.

        Parameters
        ----------
        X : list of str or Mol
            SMILES strings or RDKit Mol objects.

        Returns
        -------
        np.ndarray of shape (n_molecules, embedding_dim)
        """
        model = self._load_model()
        empty = np.zeros((0, model.message_passing.output_dim))
        if len(X) == 0:
            return empty
        mols = [
            self._featurizer(MolFromSmiles(m) if isinstance(m, str) else m)
            for m in X
        ]
        chunks = []
        for start in tqdm(
            range(0, len(mols), self.batch_size),
            desc="CheMeleon",
            unit="batch",
            leave=True,
        ):
            bmg = BatchMolGraph(mols[start : start + self.batch_size])
            bmg.to(device=model.device)
            with torch.no_grad():
                chunks.append(model.fingerprint(bmg).numpy(force=True))
        return np.concatenate(chunks, axis=0) if chunks else empty
