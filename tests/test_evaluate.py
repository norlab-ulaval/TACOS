import numpy as np
import pandas as pd

from tacos.evaluate import metrics


def test_metrics_per_run_and_pooled():
    df = pd.DataFrame({"run": ["A"] * 3 + ["B"] * 3, "label": [1.0, 2.0, 3.0] * 2,
                       "prediction": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0]})
    m = metrics(df)
    assert m.loc["A", "spearman"] == 1.0 and m.loc["B", "spearman"] == -1.0
    assert m.loc["A", "mae"] == 0.0 and np.isclose(m.loc["B", "mae"], 4 / 3)
    # pooled = all footsteps together, not the mean of the runs
    assert np.isclose(m.loc["pooled", "spearman"], df[["label", "prediction"]].corr("spearman").iloc[0, 1])
    assert np.isclose(m.loc["pooled", "mae_log"], np.abs(np.log(df.label) - np.log(df.prediction)).mean())
