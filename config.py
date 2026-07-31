"""
HSPDL configuration. Every hyperparameter locked during design discussion is
here, with a one-line note on where it came from -- nothing hidden in code.
"""

import os
from dataclasses import dataclass

DATA_ROOT = os.environ.get("HSPDL_DATA_ROOT", "/localscratch/Users/ratri/Bert/bertldl/datasets/domain_files/")
GLOBAL_FULL_PARENT = "/localscratch/Users/ratri/Bert/datasets/full_parent_sets.json"
GLOBAL_FULL_GRANDPARENT = "/localscratch/Users/ratri/Bert/datasets/full_grandparent_sets.json"
PROJECT_ROOT = os.environ.get("HSPDL_PROJECT_ROOT", "/localscratch/Users/ratri/Bert/bertldl/hspdl")

DOMAINS = ["neurology", "immunology", "embryology", "bioasq"]

# --- locked design hyperparameters ---
C_FREQ = 2.0            # omega_y_freq = 1/log(f_y + C_FREQ)
ETA1, ETA2 = 0.7, 0.3   # omega_y_hier weights: A1 (parent) vs A2 (grandparent)
BETA_DEFAULT = 0.1      # semantic-compatibility term weight (Stage 5 only); sweep {0, .05, .1, .2}
GAMMA_SEP = 0.5         # L_sep^S margin -- PROPOSED, not explicitly locked in design discussion, flag if wrong
TEMPERATURE = 0.1       # contrastive temperature, matches existing convention throughout the project

LAMBDA_ST = 0.3
LAMBDA_SSL = 0.5
LAMBDA_D = 0.3
LAMBDA_R = 0.1

# --- training strategy ---
LR_DEFAULT = 2e-5       # PROPOSED small LR (was 1e-3 before -- implicated in today's collapse).
                        # Standard transformer fine-tuning range is 1e-5 to 5e-5; flag if you want different.
EPOCHS_DEFAULT = 10
EARLY_STOP_PATIENCE = 3          # stop if loss doesn't meaningfully change for 3 consecutive epochs
EARLY_STOP_REL_THRESHOLD = 0.01  # "same loss" = <1% relative change epoch-to-epoch -- PROPOSED, flag if wrong
WARMUP_EPOCHS_DEFAULT = 2        # contrastive-only (L_ST [+ L_SSL]) epochs before BCE joins, per today's discussion

PROJ_DIM = 256
BATCH_SIZE_DEFAULT = 16  # matches what actually fit on GPU today


@dataclass
class DomainConfig:
    domain: str
    data_dir: str
    label_map_path: str
    label_freq_path: str
    legacy_parent_map_path: str
    legacy_grandparent_map_path: str
    semantic_type_path: str
    ic_scores_path: str
    omega_y_path: str

    @staticmethod
    def build(domain: str) -> "DomainConfig":
        assert domain in DOMAINS
        d = os.path.join(DATA_ROOT, domain)
        return DomainConfig(
            domain=domain,
            data_dir=d,
            label_map_path=os.path.join(d, "label_map.json"),
            label_freq_path=os.path.join(d, "label_freq.json"),
            legacy_parent_map_path=os.path.join(d, "parent_map.json"),
            legacy_grandparent_map_path=os.path.join(d, "grandparent_map.json"),
            semantic_type_path=os.path.join(d, "umls_semantic_type.json"),
            ic_scores_path=os.path.join(d, "ic_scores.json"),
            omega_y_path=os.path.join(d, "omega_y.json"),
        )


@dataclass
class TrainConfig:
    domain: str
    stage: int  # 0-5
    lr: float = LR_DEFAULT
    batch_size: int = BATCH_SIZE_DEFAULT
    epochs: int = EPOCHS_DEFAULT
    warmup_epochs: int = WARMUP_EPOCHS_DEFAULT
    early_stop_patience: int = EARLY_STOP_PATIENCE
    early_stop_rel_threshold: float = EARLY_STOP_REL_THRESHOLD
    seed: int = 42
    proj_dim: int = PROJ_DIM
    output_dir: str = ""
    lambda_st: float = LAMBDA_ST
    lambda_ssl: float = LAMBDA_SSL
    lambda_d: float = LAMBDA_D
    lambda_r: float = LAMBDA_R
    beta: float = BETA_DEFAULT

    def __post_init__(self):
        if not self.output_dir:
            self.output_dir = os.path.join(PROJECT_ROOT, "runs", f"{self.domain}_stage{self.stage}")


def active_terms_for_stage(stage: int):
    """Step 11's staged plan, exactly."""
    assert 0 <= stage <= 5
    terms = ["bce"]
    if stage >= 1:
        terms.append("st")
    if stage >= 2:
        terms.append("ssl")
    if stage >= 3:
        terms.append("dist")
    if stage >= 4:
        terms.append("sep")
    if stage == 5:
        terms.append("semantic_compat")  # optional enhancement, on top of stage 4's full set
    return terms