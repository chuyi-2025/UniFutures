#!/usr/bin/env python3
"""Train / evaluate one hierarchical ablation scheme (H01–H24)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "backtest"))

from backtest_sentiment import run_backtest  # noqa: E402
from common import class3_to_pos  # noqa: E402
from config import (  # noqa: E402
    DATA_ROOT,
    EMBED_ROOT,
    ENCODER_POOL_CANDIDATES,
    FINANCE_BERT,
    MAX_WINDOW_DOCS,
    OOS_END,
    OOS_START,
    RESULT_ROOT,
    SCHEMES,
)
from model import HierarchicalDocClassifier  # noqa: E402


def load_abandoned() -> set[str]:
    path = DATA_ROOT / "abandoned_encoders.json"
    if not path.exists():
        return set()
    return set(json.loads(path.read_text(encoding="utf-8")).keys())


def load_stream_matrix(stream: str) -> np.ndarray:
    return np.load(EMBED_ROOT / f"{stream}.npy").astype(np.float32)


def stream_dim(stream: str) -> int:
    return int(np.load(EMBED_ROOT / f"{stream}.npy", mmap_mode="r").shape[1])


def resolve_best_pools(abandoned: set[str]) -> list[str]:
    """Pick best pooling stream per alive encoder from single-model val losses."""
    best_path = RESULT_ROOT / "best_pools.json"
    if best_path.exists():
        data = json.loads(best_path.read_text(encoding="utf-8"))
        return [s for s in data.get("streams", []) if s.split("__", 1)[0] not in abandoned]

    # Fallback: final_last for each alive encoder
    streams = []
    for enc in ["emb", "finsent", "qwen06", "wiro", "fin8b"]:
        if enc in abandoned:
            continue
        cand = ENCODER_POOL_CANDIDATES[enc][0]
        if (EMBED_ROOT / f"{cand}.npy").exists():
            streams.append(cand)
    return streams


def resolve_scheme_streams(scheme_key: str, abandoned: set[str]) -> list[str]:
    scheme = SCHEMES[scheme_key]
    if scheme.resolve_best_pools:
        return resolve_best_pools(abandoned)
    out = []
    for s in scheme.streams:
        enc = s.split("__", 1)[0]
        if enc in abandoned:
            continue
        if not (EMBED_ROOT / f"{s}.npy").exists():
            continue
        out.append(s)
    return out


def streams_ready(streams: list[str]) -> bool:
    return bool(streams) and all((EMBED_ROOT / f"{s}.npy").exists() for s in streams)


class WindowDataset(Dataset):
    def __init__(
        self,
        windows: pd.DataFrame,
        matrices: list[np.ndarray],
        max_docs: int,
    ) -> None:
        self.windows = windows.reset_index(drop=True)
        self.matrices = matrices
        self.max_docs = max_docs
        self.dims = [m.shape[1] for m in matrices]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        row = self.windows.iloc[idx]
        doc_idxs = list(row["doc_idxs"])[: self.max_docs]
        ages_list = list(row["ages"])[: self.max_docs]
        n = len(doc_idxs)
        stream_tensors = []
        for m, d in zip(self.matrices, self.dims):
            arr = np.zeros((self.max_docs, d), dtype=np.float32)
            if n:
                arr[:n] = m[doc_idxs]
            stream_tensors.append(torch.from_numpy(arr))
        mask = np.zeros(self.max_docs, dtype=np.int64)
        ages = np.zeros(self.max_docs, dtype=np.int64)
        if n:
            mask[:n] = 1
            ages[:n] = np.asarray(ages_list, dtype=np.int64)
        return {
            "stream_embeds": stream_tensors,
            "attention_mask": torch.from_numpy(mask),
            "ages": torch.from_numpy(ages),
            "labels": torch.tensor(int(row["label_id"]), dtype=torch.long),
            "symbol": row["symbol"],
            "trade_date": str(pd.Timestamp(row["trade_date"]).date()),
        }


def collate(batch):
    n_streams = len(batch[0]["stream_embeds"])
    return {
        "stream_embeds": [
            torch.stack([b["stream_embeds"][i] for b in batch]) for i in range(n_streams)
        ],
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "ages": torch.stack([b["ages"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "symbol": [b["symbol"] for b in batch],
        "trade_date": [b["trade_date"] for b in batch],
    }


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    losses, preds, labels = [], [], []
    for batch in loader:
        out = model(
            stream_embeds=[x.to(device) for x in batch["stream_embeds"]],
            attention_mask=batch["attention_mask"].to(device),
            ages=batch["ages"].to(device),
            labels=batch["labels"].to(device),
        )
        losses.append(float(out.loss.item()))
        preds.append(out.logits.argmax(dim=-1).cpu().numpy())
        labels.append(batch["labels"].numpy())
    yhat = np.concatenate(preds) if preds else np.array([])
    y = np.concatenate(labels) if labels else np.array([])
    return {
        "loss": float(np.mean(losses) if losses else 0.0),
        "accuracy": float((yhat == y).mean()) if len(y) else 0.0,
    }


@torch.no_grad()
def predict_signals(model, loader, device) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in loader:
        out = model(
            stream_embeds=[x.to(device) for x in batch["stream_embeds"]],
            attention_mask=batch["attention_mask"].to(device),
            ages=batch["ages"].to(device),
        )
        probs = torch.softmax(out.logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=-1)
        for i in range(len(batch["symbol"])):
            cid = int(preds[i])
            rows.append(
                {
                    "symbol": batch["symbol"][i],
                    "trade_date": pd.Timestamp(batch["trade_date"][i]),
                    "pred_class": cid,
                    "position": class3_to_pos(cid),
                    "prob_0": float(probs[i, 0]),
                    "prob_1": float(probs[i, 1]),
                    "prob_2": float(probs[i, 2]),
                }
            )
    return pd.DataFrame(rows)


def train_one(scheme_key, streams, windows, seed, epochs, batch_size, lr, max_docs, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    scheme = SCHEMES[scheme_key]
    matrices = [load_stream_matrix(s) for s in streams]
    dims = [m.shape[1] for m in matrices]

    train_df = windows[windows["split"] == "train"]
    val_df = windows[windows["split"] == "val"]
    train_loader = DataLoader(
        WindowDataset(train_df, matrices, max_docs),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        WindowDataset(val_df, matrices, max_docs),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    model = HierarchicalDocClassifier(
        dims,
        FINANCE_BERT,
        fusion=scheme.fusion,
        bert_init=scheme.bert_init,
        use_age=scheme.use_age,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    best_state, best_val, history = None, float("inf"), []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"{scheme_key} s{seed} ep{epoch}", leave=False):
            opt.zero_grad(set_to_none=True)
            out = model(
                stream_embeds=[x.to(device) for x in batch["stream_embeds"]],
                attention_mask=batch["attention_mask"].to(device),
                ages=batch["ages"].to(device),
                labels=batch["labels"].to(device),
            )
            out.loss.backward()
            opt.step()
            losses.append(float(out.loss.item()))
        train_loss = float(np.mean(losses) if losses else 0.0)
        val_metrics = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
        print(
            f"  ep{epoch}: train={train_loss:.4f} val={val_metrics['loss']:.4f} "
            f"acc={val_metrics['accuracy']:.4f}"
        )
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "history": history,
        "best_val_loss": best_val,
        "stream_dims": dims,
        "streams": streams,
        "fusion": scheme.fusion,
        "bert_init": scheme.bert_init,
        "use_age": scheme.use_age,
        "n_train": len(train_df),
        "n_val": len(val_df),
    }


def pick_best_pools_from_singles() -> dict:
    """After H01–H12, choose lowest mean val-loss pooling per encoder."""
    abandoned = load_abandoned()
    rows = []
    for scheme_key, scheme in SCHEMES.items():
        if scheme_key > "H12" or scheme.resolve_best_pools or len(scheme.streams) != 1:
            continue
        stream = scheme.streams[0]
        enc = stream.split("__", 1)[0]
        if enc in abandoned:
            continue
        losses = []
        for seed_dir in (DATA_ROOT / "runs" / scheme_key).glob("seed_*"):
            meta = seed_dir / "meta.json"
            if not meta.exists():
                continue
            m = json.loads(meta.read_text(encoding="utf-8"))
            if m.get("status") == "ok" and m.get("best_val_loss") is not None:
                losses.append(float(m["best_val_loss"]))
        if losses:
            rows.append(
                {
                    "encoder": enc,
                    "stream": stream,
                    "scheme": scheme_key,
                    "mean_val_loss": float(np.mean(losses)),
                }
            )
    if not rows:
        return {"streams": resolve_best_pools(abandoned), "detail": []}
    df = pd.DataFrame(rows)
    chosen = []
    detail = []
    for enc, g in df.groupby("encoder"):
        best = g.sort_values("mean_val_loss").iloc[0]
        chosen.append(best["stream"])
        detail.append(best.to_dict())
    # stable encoder order
    order = ["emb", "finsent", "qwen06", "wiro", "fin8b"]
    chosen_sorted = []
    for enc in order:
        for s in chosen:
            if s.startswith(enc + "__"):
                chosen_sorted.append(s)
    out = {"streams": chosen_sorted, "detail": detail}
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "best_pools.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", required=True, choices=list(SCHEMES.keys()))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--windows", type=Path, default=DATA_ROOT / "windows_L60.parquet")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-docs", type=int, default=MAX_WINDOW_DOCS)
    ap.add_argument("--skip-backtest", action="store_true")
    ap.add_argument(
        "--resolve-best-pools",
        action="store_true",
        help="write best_pools.json from H01-H12 val losses and exit",
    )
    args = ap.parse_args()

    if args.resolve_best_pools:
        out = pick_best_pools_from_singles()
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    abandoned = load_abandoned()
    streams = resolve_scheme_streams(args.scheme, abandoned)
    run_dir = DATA_ROOT / "runs" / args.scheme / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if not streams_ready(streams):
        meta = {
            "scheme": args.scheme,
            "status": "skipped",
            "reason": "missing/abandoned streams",
            "streams": streams,
            "abandoned": sorted(abandoned),
        }
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(json.dumps(meta, indent=2))
        return

    # For H02 MRL: ensure L2 after truncate-to-512 was applied at encode time
    windows = pd.read_parquet(args.windows)
    windows["trade_date"] = pd.to_datetime(windows["trade_date"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, info = train_one(
        args.scheme,
        streams,
        windows,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_docs=args.max_docs,
        device=device,
    )
    torch.save(model.state_dict(), run_dir / "model.pt")
    (run_dir / "meta.json").write_text(
        json.dumps({"scheme": args.scheme, "seed": args.seed, "status": "ok", **info}, indent=2),
        encoding="utf-8",
    )

    matrices = [load_stream_matrix(s) for s in streams]
    loader = DataLoader(
        WindowDataset(windows, matrices, args.max_docs),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    signals = predict_signals(model, loader, device)
    signals.to_parquet(run_dir / "signals.parquet", index=False)
    signals.to_csv(run_dir / "signals.csv", index=False)

    if not args.skip_backtest:
        oos = signals[
            (signals["trade_date"] >= pd.Timestamp(OOS_START))
            & (signals["trade_date"] <= pd.Timestamp(OOS_END))
        ]
        bt_dir = RESULT_ROOT / args.scheme / f"seed_{args.seed}"
        summary = run_backtest(
            oos,
            bt_dir,
            start=pd.Timestamp(OOS_START),
            end=pd.Timestamp(OOS_END),
            hold_days=1,
        )
        med = float(summary["sharpe"].median()) if "sharpe" in summary else 0.0
        info["oos_median_sharpe"] = med
        (run_dir / "meta.json").write_text(
            json.dumps(
                {"scheme": args.scheme, "seed": args.seed, "status": "ok", **info}, indent=2
            ),
            encoding="utf-8",
        )
        print(f"[bt] {args.scheme} seed{args.seed} median_sharpe={med:.3f}")


if __name__ == "__main__":
    main()
