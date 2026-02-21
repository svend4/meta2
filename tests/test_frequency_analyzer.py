"""Тесты для puzzle_reconstruction.preprocessing.frequency_analyzer."""
import pytest
import numpy as np
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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _img(h: int = 32, w: int = 32, val: int = 128, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _img_rgb(h: int = 32, w: int = 32) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _spectrum(h: int = 32, w: int = 32) -> FreqSpectrum:
    img = _img(h, w)
    return compute_power_spectrum(img)


def _descriptor(fid: int = 0, n_bands: int = 8) -> FreqDescriptor:
    energies = [float(i + 1) for i in range(n_bands)]
    return FreqDescriptor(fragment_id=fid, band_energies=energies,
                          centroid=0.5, top_freqs=[0.1, 0.2, 0.3],
                          entropy=2.0)


# ─── TestFreqConfig ───────────────────────────────────────────────────────────

class TestFreqConfig:
    def test_defaults(self):
        cfg = FreqConfig()
        assert cfg.n_bands == 8
        assert cfg.log_scale is True
        assert cfg.normalize is True
        assert cfg.n_top_freqs == 5

    def test_valid_custom(self):
        cfg = FreqConfig(n_bands=4, log_scale=False,
                         normalize=False, n_top_freqs=3)
        assert cfg.n_bands == 4
        assert cfg.n_top_freqs == 3

    def test_n_bands_two_ok(self):
        cfg = FreqConfig(n_bands=2)
        assert cfg.n_bands == 2

    def test_n_bands_one_raises(self):
        with pytest.raises(ValueError):
            FreqConfig(n_bands=1)

    def test_n_bands_zero_raises(self):
        with pytest.raises(ValueError):
            FreqConfig(n_bands=0)

    def test_n_top_freqs_one_ok(self):
        cfg = FreqConfig(n_top_freqs=1)
        assert cfg.n_top_freqs == 1

    def test_n_top_freqs_zero_raises(self):
        with pytest.raises(ValueError):
            FreqConfig(n_top_freqs=0)

    def test_n_top_freqs_neg_raises(self):
        with pytest.raises(ValueError):
            FreqConfig(n_top_freqs=-1)


# ─── TestFreqSpectrum ─────────────────────────────────────────────────────────

class TestFreqSpectrum:
    def test_basic(self):
        sp = _spectrum()
        assert isinstance(sp, FreqSpectrum)
        assert sp.total_power >= 0.0

    def test_shape(self):
        sp = _spectrum(16, 32)
        assert sp.shape == (16, 32)

    def test_magnitude_shape(self):
        sp = _spectrum(32, 32)
        assert sp.magnitude.shape == (32, 32)

    def test_power_shape(self):
        sp = _spectrum(32, 32)
        assert sp.power.shape == (32, 32)

    def test_dc_component_non_neg(self):
        sp = _spectrum()
        assert sp.dc_component >= 0.0

    def test_total_power_neg_raises(self):
        mag = np.ones((4, 4))
        with pytest.raises(ValueError):
            FreqSpectrum(magnitude=mag, power=mag, total_power=-1.0)

    def test_total_power_zero_ok(self):
        mag = np.zeros((4, 4))
        sp = FreqSpectrum(magnitude=mag, power=mag, total_power=0.0)
        assert sp.total_power == 0.0


# ─── TestFreqDescriptor ───────────────────────────────────────────────────────

class TestFreqDescriptor:
    def test_basic(self):
        d = _descriptor(fid=3)
        assert d.fragment_id == 3
        assert len(d.band_energies) == 8

    def test_n_bands(self):
        d = _descriptor(n_bands=4)
        assert d.n_bands == 4

    def test_dominant_band(self):
        d = _descriptor(n_bands=4)
        # Energies [1, 2, 3, 4] → dominant at index 3
        assert d.dominant_band == 3

    def test_high_freq_ratio_in_range(self):
        d = _descriptor()
        assert 0.0 <= d.high_freq_ratio <= 1.0

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
        assert d.centroid == 0.0

    def test_high_freq_ratio_uniform(self):
        # Uniform energies → ratio = 0.5 for even n_bands
        energies = [1.0] * 8
        d = FreqDescriptor(fragment_id=0, band_energies=energies,
                           centroid=0.5, top_freqs=[], entropy=3.0)
        assert d.high_freq_ratio == pytest.approx(0.5)


# ─── TestComputePowerSpectrum ─────────────────────────────────────────────────

class TestComputePowerSpectrum:
    def test_returns_freq_spectrum(self):
        sp = compute_power_spectrum(_img())
        assert isinstance(sp, FreqSpectrum)

    def test_shape_matches_image(self):
        img = _img(16, 32)
        sp = compute_power_spectrum(img)
        assert sp.shape == (16, 32)

    def test_rgb_image_ok(self):
        sp = compute_power_spectrum(_img_rgb())
        assert sp.shape == (32, 32)

    def test_normalized_in_range(self):
        sp = compute_power_spectrum(_img(), normalize=True)
        assert sp.magnitude.max() <= 1.0 + 1e-6
        assert sp.magnitude.min() >= 0.0

    def test_no_normalize_large_values(self):
        img = np.full((32, 32), 200, dtype=np.uint8)
        sp = compute_power_spectrum(img, log_scale=False, normalize=False)
        assert sp.magnitude.max() > 1.0

    def test_total_power_positive(self):
        sp = compute_power_spectrum(_img())
        assert sp.total_power > 0.0

    def test_zero_image_positive_dc(self):
        # Нулевое изображение → нулевой спектр
        img = np.zeros((16, 16), dtype=np.uint8)
        sp = compute_power_spectrum(img)
        assert sp.total_power == pytest.approx(0.0)

    def test_too_small_raises(self):
        with pytest.raises(ValueError):
            compute_power_spectrum(np.zeros((1, 1)))

    def test_4d_raises(self):
        with pytest.raises(ValueError):
            compute_power_spectrum(np.zeros((4, 4, 3, 2)))

    def test_log_scale_no_neg(self):
        sp = compute_power_spectrum(_img(), log_scale=True)
        assert sp.magnitude.min() >= 0.0

    def test_constant_image(self):
        img = np.full((32, 32), 128, dtype=np.uint8)
        sp = compute_power_spectrum(img)
        # DC-компонента доминирует
        assert sp.dc_component > 0.0


# ─── TestComputeBandEnergies ──────────────────────────────────────────────────

class TestComputeBandEnergies:
    def test_returns_list(self):
        sp = _spectrum()
        bands = compute_band_energies(sp, n_bands=8)
        assert isinstance(bands, list)

    def test_length_matches_n_bands(self):
        sp = _spectrum()
        for n in (2, 4, 8, 16):
            bands = compute_band_energies(sp, n_bands=n)
            assert len(bands) == n

    def test_all_non_negative(self):
        sp = _spectrum()
        for e in compute_band_energies(sp):
            assert e >= 0.0

    def test_n_bands_one_raises(self):
        sp = _spectrum()
        with pytest.raises(ValueError):
            compute_band_energies(sp, n_bands=1)

    def test_n_bands_zero_raises(self):
        sp = _spectrum()
        with pytest.raises(ValueError):
            compute_band_energies(sp, n_bands=0)

    def test_sum_positive_for_random_img(self):
        sp = _spectrum()
        bands = compute_band_energies(sp)
        assert sum(bands) > 0.0

    def test_returns_floats(self):
        sp = _spectrum()
        for e in compute_band_energies(sp):
            assert isinstance(e, float)


# ─── TestComputeSpectralCentroid ──────────────────────────────────────────────

class TestComputeSpectralCentroid:
    def test_returns_float(self):
        sp = _spectrum()
        c = compute_spectral_centroid(sp)
        assert isinstance(c, float)

    def test_in_range(self):
        sp = _spectrum()
        c = compute_spectral_centroid(sp)
        assert 0.0 <= c <= 1.0

    def test_zero_image_centroid(self):
        img = np.zeros((16, 16), dtype=np.uint8)
        sp = compute_power_spectrum(img, log_scale=False, normalize=False)
        c = compute_spectral_centroid(sp)
        assert 0.0 <= c <= 1.0

    def test_high_freq_image_higher_centroid(self):
        # Высокочастотное изображение → centroid выше
        rng = np.random.default_rng(42)
        noisy = rng.integers(0, 256, (32, 32), dtype=np.uint8)
        smooth = np.full((32, 32), 128, dtype=np.uint8)
        c_noisy = compute_spectral_centroid(compute_power_spectrum(noisy))
        c_smooth = compute_spectral_centroid(compute_power_spectrum(smooth))
        assert c_noisy > c_smooth


# ─── TestComputeSpectralEntropy ───────────────────────────────────────────────

class TestComputeSpectralEntropy:
    def test_returns_float(self):
        sp = _spectrum()
        e = compute_spectral_entropy(sp)
        assert isinstance(e, float)

    def test_non_negative(self):
        sp = _spectrum()
        assert compute_spectral_entropy(sp) >= 0.0

    def test_constant_image_low_entropy(self):
        img = np.full((32, 32), 128, dtype=np.uint8)
        sp = compute_power_spectrum(img, log_scale=False, normalize=False)
        e = compute_spectral_entropy(sp)
        assert e >= 0.0

    def test_random_image_higher_entropy(self):
        rng = np.random.default_rng(7)
        noisy = rng.integers(0, 256, (32, 32), dtype=np.uint8)
        flat = np.full((32, 32), 128, dtype=np.uint8)
        e_noisy = compute_spectral_entropy(compute_power_spectrum(noisy))
        e_flat = compute_spectral_entropy(compute_power_spectrum(flat))
        assert e_noisy > e_flat


# ─── TestExtractTopFrequencies ────────────────────────────────────────────────

class TestExtractTopFrequencies:
    def test_returns_list(self):
        sp = _spectrum()
        freqs = extract_top_frequencies(sp, n_top=5)
        assert isinstance(freqs, list)

    def test_length_n_top(self):
        sp = _spectrum()
        for n in (1, 3, 5, 10):
            freqs = extract_top_frequencies(sp, n_top=n)
            assert len(freqs) == n

    def test_all_in_range(self):
        sp = _spectrum()
        for f in extract_top_frequencies(sp, n_top=5):
            assert 0.0 <= f <= 1.0

    def test_n_top_zero_raises(self):
        sp = _spectrum()
        with pytest.raises(ValueError):
            extract_top_frequencies(sp, n_top=0)

    def test_n_top_neg_raises(self):
        sp = _spectrum()
        with pytest.raises(ValueError):
            extract_top_frequencies(sp, n_top=-1)

    def test_returns_floats(self):
        sp = _spectrum()
        for f in extract_top_frequencies(sp):
            assert isinstance(f, float)


# ─── TestExtractFreqDescriptor ────────────────────────────────────────────────

class TestExtractFreqDescriptor:
    def test_returns_descriptor(self):
        d = extract_freq_descriptor(_img())
        assert isinstance(d, FreqDescriptor)

    def test_fragment_id_stored(self):
        d = extract_freq_descriptor(_img(), fragment_id=5)
        assert d.fragment_id == 5

    def test_band_energies_length(self):
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

    def test_entropy_non_negative(self):
        d = extract_freq_descriptor(_img())
        assert d.entropy >= 0.0

    def test_neg_fragment_id_raises(self):
        with pytest.raises(ValueError):
            extract_freq_descriptor(_img(), fragment_id=-1)

    def test_rgb_image_ok(self):
        d = extract_freq_descriptor(_img_rgb())
        assert isinstance(d, FreqDescriptor)

    def test_default_config(self):
        d = extract_freq_descriptor(_img())
        assert d.n_bands == 8

    def test_band_energies_non_negative(self):
        d = extract_freq_descriptor(_img())
        assert all(e >= 0.0 for e in d.band_energies)


# ─── TestCompareFreqDescriptors ───────────────────────────────────────────────

class TestCompareFreqDescriptors:
    def test_identical_one(self):
        d = _descriptor()
        assert compare_freq_descriptors(d, d) == pytest.approx(1.0)

    def test_in_range(self):
        a = _descriptor(fid=0)
        b = _descriptor(fid=1, n_bands=8)
        # Ensure same n_bands
        sim = compare_freq_descriptors(a, b)
        assert 0.0 <= sim <= 1.0

    def test_different_n_bands_raises(self):
        a = _descriptor(n_bands=4)
        b = _descriptor(n_bands=8)
        with pytest.raises(ValueError):
            compare_freq_descriptors(a, b)

    def test_returns_float(self):
        a = _descriptor()
        b = _descriptor()
        assert isinstance(compare_freq_descriptors(a, b), float)

    def test_same_image_high_sim(self):
        img = _img()
        da = extract_freq_descriptor(img, 0)
        db = extract_freq_descriptor(img.copy(), 1)
        assert compare_freq_descriptors(da, db) == pytest.approx(1.0, abs=1e-6)

    def test_symmetric(self):
        a = extract_freq_descriptor(_img(seed=0), 0)
        b = extract_freq_descriptor(_img(seed=5), 1)
        assert compare_freq_descriptors(a, b) == pytest.approx(
            compare_freq_descriptors(b, a), abs=1e-6)


# ─── TestBatchExtractFreqDescriptors ──────────────────────────────────────────

class TestBatchExtractFreqDescriptors:
    def test_returns_list(self):
        images = [_img(seed=i) for i in range(3)]
        result = batch_extract_freq_descriptors(images)
        assert isinstance(result, list)

    def test_length_matches(self):
        images = [_img(seed=i) for i in range(5)]
        result = batch_extract_freq_descriptors(images)
        assert len(result) == 5

    def test_fragment_ids_sequential(self):
        images = [_img(seed=i) for i in range(4)]
        result = batch_extract_freq_descriptors(images)
        for i, d in enumerate(result):
            assert d.fragment_id == i

    def test_all_are_descriptors(self):
        images = [_img(seed=i) for i in range(3)]
        for d in batch_extract_freq_descriptors(images):
            assert isinstance(d, FreqDescriptor)

    def test_empty_list(self):
        assert batch_extract_freq_descriptors([]) == []

    def test_custom_config(self):
        cfg = FreqConfig(n_bands=4)
        images = [_img(seed=i) for i in range(2)]
        result = batch_extract_freq_descriptors(images, cfg)
        for d in result:
            assert d.n_bands == 4
