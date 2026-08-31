"""Model builders used by train.py.

Only the full ego+exo method (TQN neck) is included in this release; the
single-view (ego-only / exo-only) ablation variants used a separate,
now-removed training entry point and are not wired up here.
"""


def build_head(cfg):
    head_type = getattr(cfg, "head_type", "simple").lower()

    if head_type == "simple":
        from models.head import Simple_CLS
        cls_num = cfg.num_classes - 1 if cfg.loss == "ordinal" else cfg.num_classes
        in_dim = cfg.d_model
        return Simple_CLS(in_dim, num_classes=cls_num).cuda()
    else:
        raise ValueError(f"Unsupported head type: {head_type}")


def build_neck(cfg):
    neck_type = getattr(cfg, "neck_type", "TQN")

    if neck_type == "TQN":
        from models.neck import TQN
        return TQN(cfg.d_model,
                    cfg.q_number,
                    cfg.query_var,
                    cfg.pe,
                    H=cfg.num_head,
                    Nego=cfg.num_layers_ego,
                    Nexo=cfg.num_layers_exo,
                    max_len=cfg.max_len,
                    dropout=cfg.dropout).cuda()
    else:
        raise ValueError(f"Unsupported neck type: {neck_type}")
