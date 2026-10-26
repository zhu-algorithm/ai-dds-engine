"""
DDS-Former — a domain foundation model for drug-delivery formulations.
=====================================================================

A formulation is not a molecule and not a sentence: it is an unordered set of
(excipient, functional role, amount) triples, conditioned on a dosage form and
an API, under process parameters. DDS-Former encodes exactly that structure.

Sequence layout (one formulation):

    [CLS] [FORM:<dosage_form>] [API:<api_token>]
          [EXC:e1][ROLE:r1][AMT:b1]  [EXC:e2][ROLE:r2][AMT:b2] ...
          [PROC:p1][PVAL:v1] ...
    [SEP]

Three embedding streams are summed at every position: token identity, field
type (form / api / excipient / role / amount / process), and component index
(which triple a token belongs to — so the model can bind an amount to its own
excipient while remaining permutation-invariant across components).

Pre-training: masked component modelling (MCM). Excipient identities and
amount buckets are independently masked and predicted. This forces the model
to learn excipient co-occurrence (which film former pairs with which
plasticiser) and compositional constraints (amounts summing to ~100%).

Fine-tuning: pooled [CLS] -> per-CQA regression heads with a presence mask,
so a formulation labelled only for dissolution time contributes no gradient to
the permeation head.

Distillation: `teacher_targets` carries soft labels produced by the extraction
teacher; `distill_loss` mixes them with hard labels.
"""
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ───────────────────────── field types ─────────────────────────
FIELDS = ["PAD", "CLS", "SEP", "FORM", "API", "EXC", "ROLE", "AMT", "PROC", "PVAL"]
FIELD2ID = {f: i for i, f in enumerate(FIELDS)}

# amount buckets: log-spaced over %w/w, plus a trace and a "not stated" bucket
N_AMT_BUCKETS = 24


def amount_bucket(pct):
    """%w/w -> bucket index. 0 = not stated, 1 = trace (<0.01%), 2..23 log-spaced."""
    if pct is None or (isinstance(pct, float) and math.isnan(pct)):
        return 0
    if pct <= 0.01:
        return 1
    b = int(2 + (math.log10(pct) - math.log10(0.01)) / (math.log10(100.0) - math.log10(0.01))
            * (N_AMT_BUCKETS - 3))
    return max(2, min(N_AMT_BUCKETS - 1, b))


class Vocab:
    """Separate id spaces per field, packed into one embedding table."""

    def __init__(self):
        self.tok2id, self.id2tok = {}, []
        for s in ("<pad>", "<cls>", "<sep>", "<mask>", "<unk>"):
            self.add(s)

    def add(self, tok):
        if tok not in self.tok2id:
            self.tok2id[tok] = len(self.id2tok)
            self.id2tok.append(tok)
        return self.tok2id[tok]

    def get(self, tok):
        return self.tok2id.get(tok, self.tok2id["<unk>"])

    def __len__(self):
        return len(self.id2tok)

    def save(self, path):
        json.dump(self.id2tok, open(path, "w"), ensure_ascii=False)

    @classmethod
    def load(cls, path):
        v = cls()
        v.tok2id, v.id2tok = {}, []
        for t in json.load(open(path)):
            v.add(t)
        return v

    @property
    def pad(self):
        return self.tok2id["<pad>"]

    @property
    def mask(self):
        return self.tok2id["<mask>"]


def encode(form, api, components, process=(), vocab=None, max_len=128):
    """components: [(excipient, role, pct)]; process: [(param, value_bucket)]."""
    ids, fields, comp_idx = [vocab.get("<cls>")], [FIELD2ID["CLS"]], [0]
    ids.append(vocab.get(f"FORM:{form}")); fields.append(FIELD2ID["FORM"]); comp_idx.append(0)
    ids.append(vocab.get(f"API:{api}")); fields.append(FIELD2ID["API"]); comp_idx.append(0)
    for k, (exc, role, pct) in enumerate(components, start=1):
        if len(ids) + 3 >= max_len - 1:
            break
        ids += [vocab.get(f"EXC:{exc}"), vocab.get(f"ROLE:{role}"),
                vocab.get(f"AMT:{amount_bucket(pct)}")]
        fields += [FIELD2ID["EXC"], FIELD2ID["ROLE"], FIELD2ID["AMT"]]
        comp_idx += [min(k, 31)] * 3
    for p, v in process:
        if len(ids) + 2 >= max_len - 1:
            break
        ids += [vocab.get(f"PROC:{p}"), vocab.get(f"PVAL:{v}")]
        fields += [FIELD2ID["PROC"], FIELD2ID["PVAL"]]
        comp_idx += [0, 0]
    ids.append(vocab.get("<sep>")); fields.append(FIELD2ID["SEP"]); comp_idx.append(0)
    return ids, fields, comp_idx


class DDSFormer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layer=6, n_head=8, d_ff=1024,
                 max_len=128, n_cqa=8, dropout=0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.fld = nn.Embedding(len(FIELDS), d_model)
        self.cmp = nn.Embedding(32, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model, n_head, d_ff, dropout,
                                           batch_first=True, norm_first=True,
                                           activation="gelu")
        self.enc = nn.TransformerEncoder(layer, n_layer)
        self.ln = nn.LayerNorm(d_model)
        # pre-training heads
        self.mcm_exc = nn.Linear(d_model, vocab_size)      # masked excipient identity
        self.mcm_amt = nn.Linear(d_model, N_AMT_BUCKETS)   # masked amount bucket
        # fine-tuning head: one scalar per CQA + presence-masked loss
        self.cqa = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(d_model, n_cqa))
        # token-id -> amount-bucket lookup, filled by set_amount_lut(vocab)
        self.register_buffer("amt_lut", torch.zeros(vocab_size, dtype=torch.long))

    def set_amount_lut(self, vocab):
        """Map every 'AMT:<b>' token id to its bucket index b (others stay 0)."""
        lut = torch.zeros(len(vocab), dtype=torch.long)
        for tok, i in vocab.tok2id.items():
            if tok.startswith("AMT:"):
                lut[i] = int(tok.split(":", 1)[1])
        self.amt_lut.copy_(lut.to(self.amt_lut.device))

    def _amt_target(self, token_ids):
        return self.amt_lut[token_ids]

    def backbone(self, ids, fields, comp_idx):
        B, L = ids.shape
        p = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, L)
        h = self.tok(ids) + self.fld(fields) + self.cmp(comp_idx) + self.pos(p)
        h = self.enc(self.drop(h), src_key_padding_mask=(ids == 0))
        return self.ln(h)

    def forward(self, ids, fields, comp_idx):
        h = self.backbone(ids, fields, comp_idx)
        return h, self.cqa(h[:, 0])          # pooled [CLS] -> CQA predictions


def mcm_loss(model, ids, fields, comp_idx, mask_prob=0.15, gen=None):
    """Masked component modelling over excipient-identity and amount-bucket slots."""
    exc_slot = fields == FIELD2ID["EXC"]
    amt_slot = fields == FIELD2ID["AMT"]
    cand = exc_slot | amt_slot
    r = torch.rand(ids.shape, device=ids.device, generator=gen)
    sel = cand & (r < mask_prob)
    if sel.sum() == 0:
        sel = cand & (r < r[cand].min() + 1e-9) if cand.any() else sel
    inp = ids.clone()
    inp[sel] = 3                                            # <mask>
    h = model.backbone(inp, fields, comp_idx)
    loss, n = ids.new_zeros((), dtype=torch.float), 0
    se = sel & exc_slot
    if se.any():
        loss = loss + F.cross_entropy(model.mcm_exc(h[se]), ids[se]); n += 1
    sa = sel & amt_slot
    if sa.any():
        # amount targets are bucket indices recoverable from the token string id
        tgt = model._amt_target(ids[sa])
        loss = loss + F.cross_entropy(model.mcm_amt(h[sa]), tgt); n += 1
    return loss / max(n, 1)


def cqa_loss(pred, target, present):
    """Huber on z-scored CQAs; `present` masks unlabelled entries."""
    if present.sum() == 0:
        return pred.new_zeros(())
    return F.huber_loss(pred[present], target[present], delta=1.0)


def distill_loss(student_pred, teacher_pred, target, present, alpha=0.5):
    """Mix hard labels with the teacher's soft predictions (alpha on teacher)."""
    hard = cqa_loss(student_pred, target, present)
    soft = F.mse_loss(student_pred, teacher_pred)
    return (1 - alpha) * hard + alpha * soft
