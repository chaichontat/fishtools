from fishtools.preprocess.stitching import spot_split_cut_px


def test_spot_split_cut_px_preserves_legacy_1960() -> None:
    assert spot_split_cut_px(1960) == 1024


def test_spot_split_cut_px_matches_overlap_50() -> None:
    # 1998px tiles historically used 1024px crops (overlap=50px)
    assert spot_split_cut_px(1998) == 1024

    # General rule: overlap = 2*cut - size should be 50px for typical sizes
    size = 1100
    cut = spot_split_cut_px(size)
    assert cut == 575
    assert 2 * cut - size == 50

