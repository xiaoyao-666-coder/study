"""Numeric column reader used by the frozen physical-feature joins."""
import numpy as np
def numeric(frame,columns):
    a=frame.loc[:,list(columns)].to_numpy(dtype=float,copy=True)
    if not np.isfinite(a).all():raise ValueError('Nonfinite input')
    return a
