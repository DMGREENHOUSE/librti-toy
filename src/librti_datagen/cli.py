import argparse, sys, numpy as np, pandas as pd
from pathlib import Path
from librti_datagen.models import purification, corrosion, neutroact

MODELS = {
    "purification": (purification.INPUT_COLUMNS, purification.compute),
    "corrosion": (corrosion.INPUT_COLUMNS, corrosion.compute),
    "neutroact": (neutroact.INPUT_COLUMNS, neutroact.compute),
}

def main(argv=None):
    p = argparse.ArgumentParser("librti-datagenerators")
    p.add_argument("--model", choices=MODELS.keys(), required=True)
    p.add_argument("--inputs", required=True, help="CSV of inputs matching model columns")
    p.add_argument("--output", required=True, help="CSV to write combined inputs+outputs")
    p.add_argument("--seed", type=int, default=42)
    # noise args are model-specific; accept generic flags
    p.add_argument("--noise-std", type=float, default=0.02)
    p.add_argument("--noise-pct", type=float, default=0.10)
    p.add_argument("--noise-tbr", type=float, default=0.01)
    p.add_argument("--noise-act", type=float, default=0.10)
    args = p.parse_args(argv)

    inp_cols, fn = MODELS[args.model]
    X = pd.read_csv(args.inputs)
    missing = [c for c in inp_cols if c not in X.columns]
    if missing:
        raise SystemExit(f"Missing input columns: {missing}")

    rng = np.random.default_rng(args.seed)
    kwargs = dict(noise_std=args.noise_std, noise_pct=args.noise_pct,
                  noise_tbr=args.noise_tbr, noise_act=args.noise_act)
    # filter kwargs to what the function accepts
    sig = fn.__code__.co_varnames
    kwargs = {k: v for k, v in kwargs.items() if k in sig}

    Y = fn(X, rng=rng, **kwargs)
    pd.concat([X.reset_index(drop=True), Y.reset_index(drop=True)], axis=1).to_csv(args.output, index=False)
    print(f"Wrote {args.output}")

if __name__ == "__main__":
    main()
