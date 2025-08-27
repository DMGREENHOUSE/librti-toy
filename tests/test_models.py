import numpy as np, pandas as pd
from librti_datagen.models import purification, corrosion, neutroact

def test_purification_shapes():
    X = pd.DataFrame({c: [1.0, 2.0, 3.0] for c in purification.INPUT_COLUMNS})
    rng = np.random.default_rng(0)
    Y = purification.compute(X, noise_std=0.0, rng=rng)
    assert list(Y.columns) == ["purification_efficiency"]
    assert len(Y) == len(X)
    assert (Y["purification_efficiency"].between(0.0, 1.0)).all()

# Add similar smoke tests for corrosion and neutroact
