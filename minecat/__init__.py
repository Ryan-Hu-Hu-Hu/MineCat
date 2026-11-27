"""MineCat: leakage-free prediction of enzyme kinetic parameters (K_M and k_cat).

MineCat is a three-view ensemble:

* **MineCat (tree)**        - Extremely Randomized Trees on frozen learned embeddings
                              (ProtT5, in-house SMILES Transformer, MolT5, MolGNet, EC2Vec)
                              with a pH/temperature stacking cascade.
* **MineCat (deep)**        - a multi-modal neural network that fuses eight modalities
                              (substrate graph tokens, SMILES embeddings, ESM2 sequence and
                              per-residue embeddings, EC2Vec, a chai-1 binding-pocket EGNN,
                              and pH/temperature) with bidirectional cross-attention and
                              structure-reconstruction auxiliary decoders.
* **MineCat (composition)** - Extremely Randomized Trees on hand-crafted, identity-independent
                              descriptors (AAC + CKSAAP + CTD, MACCS, Morgan).

The three views are blended in log10 space with weights chosen leave-one-fold-out on the
10-fold, 40%-sequence-identity cluster split.
"""

__version__ = "1.0.0"
