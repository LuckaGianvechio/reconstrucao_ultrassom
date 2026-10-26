"""Recover the true spatial order of MAE patch tokens.

Lives in its own dependency-free module (numpy only) because both
src/preprocessing/build_tabular_dataset.py and
src/preprocessing/embedding_pooling.py need it, and importing it from either of
those would close an import cycle
(build_tabular_dataset -> embedding_pooling -> build_tabular_dataset_vae ->
build_tabular_dataset).

WHY THIS IS NEEDED. MAE's `random_masking` shuffles the patch sequence with
`argsort(rand)` even at mask_ratio=0.0 (see mae/models_mae.py), and the shuffle
is drawn PER IMAGE. So token index `i` in embeddings_per_patch.json is a
different spatial patch in every image. `serve/generate_embeddings.py` stores the
`ids_restore` permutation that undoes it; anything treating a token index as a
spatial position must apply it first.

Mean- and CLS-pooling are exempt: the mean over patches is permutation-invariant,
and the CLS token is prepended AFTER masking, so it is always index 0.
"""
import numpy as np


def restore_spatial_order(tokens: np.ndarray, ids_restore) -> np.ndarray:
    """Reorder a (197, 768) token tensor from MAE's shuffled sequence order into
    true spatial order (row-major over the 14x14 patch grid).

    `tokens[0]` is the CLS token and is left in place - MAE appends it after
    masking, so it is never part of the shuffle. The 196 patch tokens are gathered
    with `ids_restore`, exactly as MAE's own `forward_decoder` does.

    Decoding the result with `decode_embedding(..., ids_restore=None)` is
    byte-identical to decoding the shuffled tokens with their real `ids_restore`,
    since the decoder then gathers with arange (the identity).
    """
    if ids_restore is None:
        raise ValueError(
            'embedding record has no ids_restore, so spatial order cannot be recovered; '
            'regenerate embeddings with serve/generate_embeddings.py')
    ids = np.asarray(ids_restore, dtype=np.int64).ravel()
    return np.concatenate([tokens[:1], tokens[1:][ids]], axis=0)
