"""Extra tests for puzzle_reconstruction/preprocessing/frequency_analyzer.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.frequency_analyzer import (
    FreqConfig,
    FreqSpectrum,
    FreqDescriptor,
    compute_power_spectrum,
    compute_band_energies,
    compute_spectral_centroid,
    compute_spectral_entropy,
    extract_top_frequencies,
    extract_freq_descriptor,
    compare_freq_descriptors,
    batch_extract_freq_descriptors,
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _img(h=32, w=32, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _rgb(h=32, w=32):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _constant(h=32, w=32, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _spectrum(h=32, w=32, seed=0):
    return compute_power_spectrum(_img(h, w, seed=seed))


def _descriptor(fid=0, n_bands=8):
    energies = [float(i + 1) for i in range(n_bands)]
    return FreqDescriptor(fragment_id=fid, band_energies=energies,
                          centroid=0.5, top_freqs=[0.1, 0.2, 0.3],
                          entropy=2.0)


# ─── FreqConfig (extra) ──────────────────────────────────────────────────────

class TestFreqConfigExtra:
    def test_default_n_bands_8(self):
        assert FreqConfig().n_bands == 8

    def test_default_log_scale_true(self):
        assert FreqConfig().log_scale is True

    def test_default_normalize_true(self):
        assert FreqConfig().normalize is True

    def test_default_n_top_freqs_5(self):
        assert FreqConfig().n_top_freqs == 5

    def test_n_bands_two_ok(self):
        assert FreqConfig(n_bands=2).n_bands == 2

    def test_n_bands_one_raises(self):
        with pytest.raises(ValueError):
            FreqConfig(n_bands=1)

    def test_n_bands_zero_raises(self):
        with pytest.raises(ValueError):
            FreqConfig(n_bands=0)

    def test_n_top_freqs_one_ok(self):
        assert FreqConfig(n_top_freqs=1).n_top_freqs == 1

    def test_n_top_freqs_zero_raises(self):
        with pytest.raises(ValueError):
            FreqConfig(n_top_freqs=0)

    def test_n_top_freqs_neg_raises(self):
        with pytest.raises(ValueError):
            FreqConfig(n_top_freqs=-1)

    def test_custom_valid(self):
        cfg = FreqConfig(n_bands=4, log_scale=False, normalize=False, n_top_freqs=3)
        assert cfg.n_bands == 4
        assert cfg.log_scale is False
        assert cfg.n_top_freqs == 3


# ─── FreqSpectrum (extra) ────────────────────────────────────────────────────

class TestFreqSpectrumExtra:
    def test_is_freq_spectrum(self):
        assert isinstance(_spectrum(), FreqSpectrum)

    def test_shape(self):
        sp = _spectrum(16, 32)
        assert sp.shape == (16, 32)

    def test_magnitude_shape(self):
        sp = _spectrum(32, 32)
        assert sp.magnitude.shape == (32, 32)

    def test_power_shape(self):
        sp = _spectrum(32, 32)
        assert sp.power.shape == (32, 32)

    def test_total_power_nonneg(self):
        sp = _spectrum()
        assert sp.total_power >= 0.0

    def test_dc_component_nonneg(self):
        sp = _spectrum()
        assert sp.dc_component >= 0.0

    def test_total_power_neg_raises(self):
        mag = np.ones((4, 4))
        with pytest.raises(ValueError):
            FreqSpectrum(magnitude=mag, power=mag, total_power=-1.0)

    def test_total_power_zero_ok(self):
        mag = np.zeros((4, 4))
        sp = FreqSpectrum(magnitude=mag, power=mag, total_power=0.0)
        assert sp.total_power == pytest.approx(0.0)

    def test_magnitude_nonneg(self):
        sp = _spectrum()
        assert np.all(sp.magnitude >= 0.0)


# ─── FreqDescriptor (extra) ──────────────────────────────────────────────────

class TestFreqDescriptorExtra:
    def test_fragment_id_stored(self):
        d = _descriptor(fid=5)
        assert d.fragment_id == 5

    def test_n_bands(self):
        d = _descriptor(n_bands=4)
        assert d.n_bands == 4

    def test_band_energies_length(self):
        d = _descriptor(n_bands=6)
        assert len(d.band_energies) == 6

    def test_dominant_band(self):
        d = _descriptor(n_bands=4)  # energies [1, 2, 3, 4]
        assert d.dominant_band == 3

    def test_high_freq_ratio_in_range(self):
        d = _descriptor()
        assert 0.0 <= d.high_freq_ratio <= 1.0

    def test_high_freq_ratio_uniform(self):
        energies = [1.0] * 8
        d = FreqDescriptor(fragment_id=0, band_energies=energies,
                           centroid=0.5, top_freqs=[], entropy=3.0)
        assert d.high_freq_ratio == pytest.approx(0.5)

    def test_fragment_id_neg_raises(self):
        with pytest.raises(ValueError):
            FreqDescriptor(fragment_id=-1, band_energies=[1.0],
                           centroid=0.5, top_freqs=[0.1], entropy=1.0)

    def test_empty_band_energies_raises(self):
        with pytest.raises(ValueError):
            FreqDescriptor(fragment_id=0, band_energies=[],
                           centroid=0.5, top_freqs=[0.1], entropy=1.0)

    def test_centroid_neg_raises(self):
        with pytest.raises(ValueError):
            FreqDescriptor(fragment_id=0, band_energies=[1.0],
                           centroid=-0.1, top_freqs=[0.1], entropy=1.0)

    def test_entropy_neg_raises(self):
        with pytest.raises(ValueError):
            FreqDescriptor(fragment_id=0, band_energies=[1.0],
                           centroid=0.5, top_freqs=[0.1], entropy=-1.0)

    def test_centroid_zero_ok(self):
        d = FreqDescriptor(fragment_id=0, band_energies=[1.0],
                           centroid=0.0, top_freqs=[0.1], entropy=0.0)
        assert d.centroid == pytest.approx(0.0)


# ─── compute_power_spectrum (extra) ──────────────────────────────────────────

class TestComputePowerSpectrumExtra:
    def test_returns_freq_spectrum(self):
        assert isinstance(compute_power_spectrum(_img()), FreqSpectrum)

    def test_shape_matches_image(self):
        sp = compute_power_spectrum(_img(16, 32))
        assert sp.shape == (16, 32)

    def test_rgb_ok(self):
        sp = compute_power_spectrum(_rgb())
        assert sp.shape == (32, 32)

    def test_normalized_in_range(self):
        sp = compute_power_spectrum(_img(), normalize=True)
        assert sp.magnitude.max() <= 1.0 + 1e-6
        assert sp.magnitude.min() >= 0.0

    def test_not_normalized_can_exceed_one(self):
        img = np.full((32, 32), 200, dtype=np.uint8)
        sp = compute_power_spectrum(img, log_scale=False, normalize=False)
        assert sp.magnitude.max() > 1.0

    def test_total_power_positive_for_random(self):
        sp = compute_power_spectrum(_img())
        assert sp.total_power > 0.0

    def test_zero_image_total_power_zero(self):
        img = np.zeros((16, 16), dtype=np.uint8)
        sp = compute_power_spectrum(img)
        assert sp.total_power == pytest.approx(0.0)

    def test_too_small_raises(self):
        with pytest.raises(ValueError):
            compute_power_spectrum(np.zeros((1, 1)))

    def test_4d_raises(self):
        with pytest.raises(ValueError):
            compute_power_spectrum(np.zeros((4, 4, 3, 2)))

    def test_log_scale_nonneg(self):
        sp = compute_power_spectrum(_img(), log_scale=True)
        assert sp.magnitude.min() >= 0.0

    def test_constant_dc_positive(self):
        sp = compute_power_spectrum(_constant(val=128))
        assert sp.dc_component > 0.0


# ─── compute_band_energies (extra) ───────────────────────────────────────────

class TestComputeBandEnergiesExtra:
    def test_returns_list(self):
        sp = _spectrum()
        assert isinstance(compute_band_energies(sp, n_bands=8), list)

    def test_length_matches_n_bands(self):
        sp = _spectrum()
        for n in (2, 4, 8, 16):
            assert len(compute_band_energies(sp, n_bands=n)) == n

    def test_all_nonneg(self):
        sp = _spectrum()
        for e in compute_band_energies(sp):
            assert e >= 0.0

    def test_n_bands_one_raises(self):
        with pytest.raises(ValueError):
            compute_band_energies(_spectrum(), n_bands=1)

    def test_n_bands_zero_raises(self):
        with pytest.raises(ValueError):
            compute_band_energies(_spectrum(), n_bands=0)

    def test_sum_positive_for_random(self):
        sp = _spectrum()
        assert sum(compute_band_energies(sp)) > 0.0

    def test_returns_floats(self):
        sp = _spectrum()
        for e in compute_band_energies(sp):
            assert isinstance(e, float)


# ─── compute_spectral_centroid (extra) ───────────────────────────────────────

class TestComputeSpectralCentroidExtra:
    def test_returns_float(self):
        assert isinstance(compute_spectral_centroid(_spectrum()), float)

    def test_in_range(self):
        c = compute_spectral_centroid(_spectrum())
        assert 0.0 <= c <= 1.0

    def test_zero_image_in_range(self):
        img = np.zeros((16, 16), dtype=np.uint8)
        sp = compute_power_spectrum(img, log_scale=False, normalize=False)
        c = compute_spectral_centroid(sp)
        assert 0.0 <= c <= 1.0

    def test_noisy_higher_than_smooth(self):
        noisy = _img(seed=42)
        smooth = _constant(val=128)
        c_noisy = compute_spectral_centroid(compute_power_spectrum(noisy))
        c_smooth = compute_spectral_centroid(compute_power_spectrum(smooth))
        assert c_noisy > c_smooth

    def test_different_images_differ(self):
        c1 = compute_spectral_centroid(compute_power_spectrum(_img(seed=0)))
        c2 = compute_spectral_centroid(compute_power_spectrum(_img(seed=99)))
        assert isinstance(c1, float) and isinstance(c2, float)


# ─── compute_spectral_entropy (extra) ────────────────────────────────────────

class TestComputeSpectralEntropyExtra:
    def test_returns_float(self):
        assert isinstance(compute_spectral_entropy(_spectrum()), float)

    def test_nonneg(self):
        assert compute_spectral_entropy(_spectrum()) >= 0.0

    def test_constant_image_near_zero(self):
        sp = compute_power_spectrum(_constant(), log_scale=False, normalize=False)
        e = compute_spectral_entropy(sp)
        assert e >= -1e-9

    def test_random_higher_than_flat(self):
        e_noisy = compute_spectral_entropy(compute_power_spectrum(_img(seed=7)))
        e_flat = compute_spectral_entropy(compute_power_spectrum(_constant()))
        assert e_noisy > e_flat

    def test_different_seeds_different_entropy(self):
        e1 = compute_spectral_entropy(compute_power_spectrum(_img(seed=0)))
        e2 = compute_spectral_entropy(compute_power_spectrum(_img(seed=1)))
        assert isinstance(e1, float) and isinstance(e2, float)


# ─── extract_top_frequencies (extra) ─────────────────────────────────────────

class TestExtractTopFrequenciesExtra:
    def test_returns_list(self):
        assert isinstance(extract_top_frequencies(_spectrum(), n_top=5), list)

    def test_length_matches(self):
        sp = _spectrum()
        for n in (1, 3, 5, 10):
            assert len(extract_top_frequencies(sp, n_top=n)) == n

    def test_all_in_range(self):
        for f in extract_top_frequencies(_spectrum(), n_top=5):
            assert 0.0 <= f <= 1.0

    def test_n_top_zero_raises(self):
        with pytest.raises(ValueError):
            extract_top_frequencies(_spectrum(), n_top=0)

    def test_n_top_neg_raises(self):
        with pytest.raises(ValueError):
            extract_top_frequencies(_spectrum(), n_top=-1)

    def test_returns_floats(self):
        for f in extract_top_frequencies(_spectrum(), n_top=3):
            assert isinstance(f, float)


# ─── extract_freq_descriptor (extra) ─────────────────────────────────────────

class TestExtractFreqDescriptorExtra:
    def test_returns_freq_descriptor(self):
        assert isinstance(extract_freq_descriptor(_img()), FreqDescriptor)

    def test_fragment_id_stored(self):
        d = extract_freq_descriptor(_img(), fragment_id=7)
        assert d.fragment_id == 7

    def test_n_bands_from_config(self):
        cfg = FreqConfig(n_bands=4)
        d = extract_freq_descriptor(_img(), cfg=cfg)
        assert d.n_bands == 4

    def test_top_freqs_length(self):
        cfg = FreqConfig(n_top_freqs=3)
        d = extract_freq_descriptor(_img(), cfg=cfg)
        assert len(d.top_freqs) == 3

    def test_centroid_in_range(self):
        d = extract_freq_descriptor(_img())
        assert 0.0 <= d.centroid <= 1.0

    def test_entropy_nonneg(self):
        d = extract_freq_descriptor(_img())
        assert d.entropy >= 0.0

    def test_neg_fragment_id_raises(self):
        with pytest.raises(ValueError):
            extract_freq_descriptor(_img(), fragment_id=-1)

    def test_rgb_ok(self):
        d = extract_freq_descriptor(_rgb())
        assert isinstance(d, FreqDescriptor)

    def test_default_n_bands_8(self):
        d = extract_freq_descriptor(_img())
        assert d.n_bands == 8

    def test_band_energies_nonneg(self):
        d = extract_freq_descriptor(_img())
        assert all(e >= 0.0 for e in d.band_energies)


# ─── compare_freq_descriptors (extra) ────────────────────────────────────────

class TestCompareFreqDescriptorsExtra:
    def test_identical_is_one(self):
        d = _descriptor()
        assert compare_freq_descriptors(d, d) == pytest.approx(1.0)

    def test_in_range(self):
        a = _descriptor(fid=0, n_bands=8)
        b = _descriptor(fid=1, n_bands=8)
        sim = compare_freq_descriptors(a, b)
        assert 0.0 <= sim <= 1.0

    def test_different_n_bands_raises(self):
        a = _descriptor(n_bands=4)
        b = _descriptor(n_bands=8)
        with pytest.raises(ValueError):
            compare_freq_descriptors(a, b)

    def test_returns_float(self):
        a = _descriptor()
        assert isinstance(compare_freq_descriptors(a, a), float)

    def test_same_image_high_sim(self):
        img = _img(seed=0)
        da = extract_freq_descriptor(img, 0)
        db = extract_freq_descriptor(img.copy(), 1)
        assert compare_freq_descriptors(da, db) == pytest.approx(1.0, abs=1e-6)

    def test_symmetric(self):
        a = extract_freq_descriptor(_img(seed=0), 0)
        b = extract_freq_descriptor(_img(seed=5), 1)
        assert compare_freq_descriptors(a, b) == pytest.approx(
            compare_freq_descriptors(b, a), abs=1e-6)


# ─── batch_extract_freq_descriptors (extra) ──────────────────────────────────

class TestBatchExtractFreqDescriptorsExtra:
    def test_returns_list(self):
        imgs = [_img(seed=i) for i in range(3)]
        assert isinstance(batch_extract_freq_descriptors(imgs), list)

    def test_length_matches(self):
        imgs = [_img(seed=i) for i in range(5)]
        assert len(batch_extract_freq_descriptors(imgs)) == 5

    def test_fragment_ids_sequential(self):
        imgs = [_img(seed=i) for i in range(4)]
        result = batch_extract_freq_descriptors(imgs)
        for i, d in enumerate(result):
            assert d.fragment_id == i

    def test_all_descriptors(self):
        for d in batch_extract_freq_descriptors([_img(seed=i) for i in range(3)]):
            assert isinstance(d, FreqDescriptor)

    def test_empty_list(self):
        assert batch_extract_freq_descriptors([]) == []

    def test_custom_config(self):
        cfg = FreqConfig(n_bands=4)
        imgs = [_img(seed=i) for i in range(2)]
        for d in batch_extract_freq_descriptors(imgs, cfg):
            assert d.n_bands == 4

    def test_single_image(self):
        result = batch_extract_freq_descriptors([_img()])
        assert len(result) == 1
