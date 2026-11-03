import pytest

from baidu_reverse.captcha.browser import cookies_to_header, find_browser
from baidu_reverse.captcha.solver import _human_curve, _parse_rotation, percent_for


def test_percent_for_matches_the_sdk_rotation_formula():
    # SDK: css rotation = start + 360 * percent (clockwise). We need it == angle (mod 360).
    assert percent_for(97.3) == 0.27
    assert percent_for(180) == 0.5
    assert percent_for(90, start_deg=30) == pytest.approx(0.17)           # 30 + 0.17*360 = 91.2
    assert percent_for(10, start_deg=90) == pytest.approx(0.78)           # wraps around (10-90 = -80 -> 280)
    assert 0.0 <= percent_for(359.9) < 1.0                                # 1.0 would be a full turn == 0


def test_percent_for_roundtrip_error_is_within_the_sdk_quantisation():
    for ang in range(0, 360, 7):
        for start in (0.0, 45.0, 200.0):
            p = percent_for(ang, start)
            err = ((start + 360 * p) - ang + 180) % 360 - 180
            assert abs(err) <= 1.81, (ang, start, err)                    # 2 decimals -> 3.6 deg steps -> 1.8 max


def test_parse_rotation():
    assert _parse_rotation("z-index: 2;transform: rotate(93.6deg);") == 93.6
    assert _parse_rotation("transform: rotate(-12.5deg)") == -12.5
    assert _parse_rotation("") == 0.0 and _parse_rotation(None) == 0.0


def test_human_curve_is_a_smooth_monotonic_0_to_1():
    xs = [_human_curve(i / 50) for i in range(51)]
    assert xs[0] == 0 and xs[-1] == 1 and all(b >= a for a, b in zip(xs, xs[1:]))
    assert xs[1] < 0.01 and xs[-2] > 0.99                                  # eases in and out


def test_cookies_to_header_keeps_baidu_domains_only():
    header = cookies_to_header([
        {"name": "BAIDUID", "value": "a", "domain": ".baidu.com"},
        {"name": "BA_HECTOR", "value": "b", "domain": "chat.baidu.com"},
        {"name": "tracker", "value": "x", "domain": ".evil-baidu.com.example"},
        {"name": "", "value": "nameless", "domain": ".baidu.com"},
        {"name": "other", "value": "y", "domain": ".google.com"},
        {"name": "BAIDUID", "value": "a2", "domain": "wenxin.baidu.com"},   # later duplicate wins
    ])
    assert header == "BAIDUID=a2; BA_HECTOR=b"


def test_find_browser_honours_env_and_explicit_path(tmp_path, monkeypatch):
    fake = tmp_path / "brave.exe"
    fake.write_text("")
    assert find_browser(str(fake)) == str(fake)
    monkeypatch.setenv("BAIDU_REVERSE_BROWSER", str(fake))
    assert find_browser() == str(fake)


def test_angle_predictor_preprocessing_shape_and_tta_shift():
    """No weights needed: check the preprocessing and the TTA un-shifting arithmetic."""
    torch = pytest.importorskip("torch")  # per-test: the rest of this module needs no ML stack
    pytest.importorskip("torchvision")
    pytest.importorskip("PIL")
    import numpy as np  # noqa: F401
    from PIL import Image

    from baidu_reverse.captcha.angle import CLS_NUM, TARGET_SIZE, AnglePredictor

    ap = AnglePredictor.__new__(AnglePredictor)          # skip loading the 70 MB model
    ap._torch = torch
    img = Image.new("RGB", (350, 350), (120, 40, 200))
    t = ap._prep(img)
    assert tuple(t.shape) == (3, TARGET_SIZE, TARGET_SIZE) and t.dtype == torch.float32

    # fake model that "sees" an image rotated CCW by true+shift: argmax bin = (true+shift)
    true_bin = 27

    class Fake:
        def __call__(self, batch):
            out = torch.zeros(batch.shape[0], CLS_NUM)
            for i, shift in enumerate((0, 90, 180, 270)[: batch.shape[0]]):
                out[i, (true_bin + shift * CLS_NUM // 360) % CLS_NUM] = 20.0
            return out

    ap.model, ap.device = Fake(), torch.device("cpu")
    angle, conf = ap.predict(img, tta=True)
    assert abs(angle - true_bin * 360 / CLS_NUM) < 1.0 and conf > 0.99
