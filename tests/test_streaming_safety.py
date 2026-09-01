"""The live audio callback must never crash or emit non-finite output, even
if the model itself fails - see StreamingEnhancer.process_chunk's docstring.
"""
import numpy as np
import pytest

from src.framing import HOP
from src.stream_demo import StreamingEnhancer

ONNX = "artifacts/model_lowsnr_simple.onnx"


def _has_model():
    from pathlib import Path
    return Path(ONNX).exists()


pytestmark = pytest.mark.skipif(not _has_model(), reason=f"{ONNX} not present")


def test_process_chunk_survives_a_broken_session():
    enh = StreamingEnhancer(ONNX)
    chunk = (np.random.default_rng(0).standard_normal(HOP) * 0.1).astype(np.float32)

    def _boom(*a, **kw):
        raise RuntimeError("simulated ONNX Runtime failure")
    enh.sess.run = _boom

    out = enh.process_chunk(chunk)

    assert np.all(np.isfinite(out))
    assert np.allclose(out, np.clip(chunk, -1.0, 1.0))
    assert enh.consecutive_failures == 1


def test_process_chunk_survives_nonfinite_model_output():
    enh = StreamingEnhancer(ONNX)
    chunk = (np.random.default_rng(1).standard_normal(HOP) * 0.1).astype(np.float32)

    def _nan_output(*a, **kw):
        bad = np.full((1, 257, 1, 2), np.nan, dtype=np.float32)
        return bad, enh.conv, enh.tra, enh.inter
    enh.sess.run = _nan_output

    out = enh.process_chunk(chunk)

    assert np.all(np.isfinite(out))
    assert np.allclose(out, np.clip(chunk, -1.0, 1.0))


def test_process_chunk_recovers_on_the_next_good_frame():
    enh = StreamingEnhancer(ONNX)
    rng = np.random.default_rng(2)
    chunk = (rng.standard_normal(HOP) * 0.1).astype(np.float32)

    real_run = enh.sess.run
    enh.sess.run = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom"))
    enh.process_chunk(chunk)  # fails once, resets state
    assert enh.consecutive_failures == 1

    enh.sess.run = real_run
    out = enh.process_chunk((rng.standard_normal(HOP) * 0.1).astype(np.float32))

    assert np.all(np.isfinite(out))
    assert enh.consecutive_failures == 0
