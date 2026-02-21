"""Тесты для puzzle_reconstruction.preprocessing.frequency_analyzer."""
import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.frequency_analyzer import (
    FreqConfig,
    FreqDescriptor,
    FreqSpectrum,
    batch_extract_freq_descriptors,
    compare_freq_descriptors,
    compute_band_energies,
    compute_power_spectrum,
    compute_spectral_centroid,
    compute_spectral_entropy,
    extract_freq_descriptor,
    extract_top_frequencies,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=32, w=32, seed=0):
    return np.random.default_rng(seed).integers(0, 255, (h, w),
                                                dtype=np.uint8)


def _rgb(h=32, w=32, seed=1):
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3),
                                                dtype=np.uint8)


# ─── TestFreqConfig ───────────────────────────────────────────────────────────

class TestFreqConfig:
    def test_defaults(self):
        cfg = FreqConfig()
        assert cfg.n_bands == 8
        assert cfg.log_scale is True
        assert cfg.normalize is True
        assert cfg.n_top_freqs == 5

    def test_n_bands_lt_2_raises(self):
        with pytest.raises(ValueError):
            FreqConfig(n_bands=1)

    def test_n_top_freqs_lt_1_raises(self):
        with pytest.raises(ValueError):
            FreqConfig(n_top_freqs=0)

    def test_valid_custom(self):
        cfg = FreqConfig(n_bands=4, log_scale=False, normalize=False, n_top_freqs=3)
        assert cfg.n_bands == 4
        assert cfg.n_top_freqs == 3


# ─── TestFreqSpectrum ─────────────────────────────────────────────────────────

class TestFreqSpectrum:
    def _make(self, h=8, w=8):
        mag = np.ones((h, w), dtype=float)
        pwr = mag ** 2
        return FreqSpectrum(magnitude=mag, power=pwr, total_power=float(pwr.sum()))

    def test_basic_construction(self):
        s = self._make()
        assert s.total_power > 0.0

    def test_negative_total_power_raises(self):
        mag = np.ones((4, 4))
        with pytest.raises(ValueError):
            FreqSpectrum(magnitude=mag, power=mag, total_power=-1.0)

    def test_shape_property(self):
        s = self._make(h=6, w=10)
        assert s.shape == (6, 10)

    def test_dc_component_at_center(self):
        pwr = np.zeros((8, 8))
        pwr[4, 4] = 100.0  # center
        s = FreqSpectrum(magnitude=np.sqrt(pwr), power=pwr, total_power=100.0)
        assert abs(s.dc_component - 100.0) < 1e-9


# ─── TestFreqDescriptor ───────────────────────────────────────────────────────

class TestFreqDescriptor:
    def _make(self, fid=0, bands=None, centroid=0.3, entropy=4.0):
        if bands is None:
            bands = [1.0] * 8
        return FreqDescriptor(fragment_id=fid, band_energies=bands,
                              centroid=centroid, top_freqs=[0.1, 0.2],
                              entropy=entropy)

    def test_basic_construction(self):
        d = self._make()
        assert d.fragment_id == 0

    def test_negative_id_raises(self):
        with pytest.raises(ValueError):
            FreqDescriptor(fragment_id=-1, band_energies=[1.0],
                           centroid=0.0, top_freqs=[], entropy=0.0)

    def test_empty_band_energies_raises(self):
        with pytest.raises(ValueError):
            FreqDescriptor(fragment_id=0, band_energies=[],
                           centroid=0.0, top_freqs=[], entropy=0.0)

    def test_negative_centroid_raises(self):
        with pytest.raises(ValueError):
            FreqDescriptor(fragment_id=0, band_energies=[1.0],
                           centroid=-0.1, top_freqs=[], entropy=0.0)

    def test_negative_entropy_raises(self):
        with pytest.raises(ValueError):
            FreqDescriptor(fragment_id=0, band_energies=[1.0],
                           centroid=0.0, top_freqs=[], entropy=-0.1)

    def test_n_bands_property(self):
        d = self._make(bands=[1.0] * 6)
        assert d.n_bands == 6

    def test_dominant_band_property(self):
        bands = [0.1, 0.1, 5.0, 0.1, 0.1, 0.1, 0.1, 0.1]
        d = self._make(bands=bands)
        assert d.dominant_band == 2

    def test_high_freq_ratio_property(self):
        bands = [0.0] * 4 + [1.0] * 4  # all energy in upper half
        d = self._make(bands=bands)
        assert abs(d.high_freq_ratio - 1.0) < 1e-9

    def test_high_freq_ratio_zero_for_low_freqs(self):
        bands = [1.0] * 4 + [0.0] * 4
        d = self._make(bands=bands)
        assert abs(d.high_freq_ratio - 0.0) < 1e-9


# ─── TestComputePowerSpectrum ─────────────────────────────────────────────────

class TestComputePowerSpectrum:
    def test_returns_freq_spectrum(self):
        s = compute_power_spectrum(_gray())
        assert isinstance(s, FreqSpectrum)

    def test_shape_matches_input(self):
        img = _gray(32, 48)
        s = compute_power_spectrum(img)
        assert s.shape == (32, 48)

    def test_rgb_accepted(self):
        s = compute_power_spectrum(_rgb())
        assert isinstance(s, FreqSpectrum)

    def test_total_power_positive(self):
        s = compute_power_spectrum(_gray())
        assert s.total_power > 0.0

    def test_normalize_flag_range(self):
        s = compute_power_spectrum(_gray(), normalize=True)
        assert s.magnitude.max() <= 1.0

    def test_log_scale_false(self):
        s = compute_power_spectrum(_gray(), log_scale=False, normalize=False)
        assert isinstance(s, FreqSpectrum)

    def test_too_small_image_raises(self):
        with pytest.raises(ValueError):
            compute_power_spectrum(np.array([[1]]))

    def test_4d_image_raises(self):
        with pytest.raises(ValueError):
            compute_power_spectrum(np.zeros((2, 2, 3, 2)))


# ─── TestComputeBandEnergies ──────────────────────────────────────────────────

class TestComputeBandEnergies:
    def _spectrum(self):
        return compute_power_spectrum(_gray())

    def test_returns_list(self):
        be = compute_band_energies(self._spectrum(), n_bands=8)
        assert isinstance(be, list)

    def test_length_equals_n_bands(self):
        be = compute_band_energies(self._spectrum(), n_bands=6)
        assert len(be) == 6

    def test_all_nonnegative(self):
        be = compute_band_energies(self._spectrum(), n_bands=8)
        assert all(e >= 0.0 for e in be)

    def test_n_bands_lt_2_raises(self):
        with pytest.raises(ValueError):
            compute_band_energies(self._spectrum(), n_bands=1)

    def test_total_energy_positive(self):
        be = compute_band_energies(self._spectrum(), n_bands=8)
        assert sum(be) > 0.0


# ─── TestComputeSpectralCentroid ──────────────────────────────────────────────

class TestComputeSpectralCentroid:
    def test_returns_float(self):
        c = compute_spectral_centroid(compute_power_spectrum(_gray()))
        assert isinstance(c, float)

    def test_in_0_1(self):
        c = compute_spectral_centroid(compute_power_spectrum(_gray()))
        assert 0.0 <= c <= 1.0

    def test_dc_only_zero_centroid(self):
        # pure DC: all power at center → centroid near 0
        pwr = np.zeros((32, 32))
        pwr[16, 16] = 1000.0
        s = FreqSpectrum(magnitude=np.sqrt(pwr), power=pwr,
                         total_power=float(pwr.sum()))
        c = compute_spectral_centroid(s)
        assert c < 0.1


# ─── TestComputeSpectralEntropy ───────────────────────────────────────────────

class TestComputeSpectralEntropy:
    def test_returns_float(self):
        e = compute_spectral_entropy(compute_power_spectrum(_gray()))
        assert isinstance(e, float)

    def test_nonnegative(self):
        e = compute_spectral_entropy(compute_power_spectrum(_gray()))
        assert e >= 0.0

    def test_uniform_power_high_entropy(self):
        pwr = np.ones((32, 32))
        s = FreqSpectrum(magnitude=pwr, power=pwr,
                         total_power=float(pwr.sum()))
        e_uniform = compute_spectral_entropy(s)
        # Concentrated power → lower entropy
        pwr2 = np.zeros((32, 32))
        pwr2[0, 0] = 1024.0
        s2 = FreqSpectrum(magnitude=np.sqrt(pwr2), power=pwr2,
                          total_power=float(pwr2.sum()))
        e_conc = compute_spectral_entropy(s2)
        assert e_uniform > e_conc


# ─── TestExtractTopFrequencies ────────────────────────────────────────────────

class TestExtractTopFrequencies:
    def test_returns_list(self):
        s = compute_power_spectrum(_gray())
        f = extract_top_frequencies(s, n_top=5)
        assert isinstance(f, list)

    def test_length_equals_n_top(self):
        s = compute_power_spectrum(_gray())
        f = extract_top_frequencies(s, n_top=3)
        assert len(f) == 3

    def test_all_in_0_1(self):
        s = compute_power_spectrum(_gray())
        f = extract_top_frequencies(s, n_top=5)
        assert all(0.0 <= v <= 1.0 for v in f)

    def test_n_top_lt_1_raises(self):
        s = compute_power_spectrum(_gray())
        with pytest.raises(ValueError):
            extract_top_frequencies(s, n_top=0)


# ─── TestExtractFreqDescriptor ────────────────────────────────────────────────

class TestExtractFreqDescriptor:
    def test_returns_freq_descriptor(self):
        d = extract_freq_descriptor(_gray())
        assert isinstance(d, FreqDescriptor)

    def test_fragment_id_set(self):
        d = extract_freq_descriptor(_gray(), fragment_id=5)
        assert d.fragment_id == 5

    def test_n_bands_matches_config(self):
        cfg = FreqConfig(n_bands=6)
        d = extract_freq_descriptor(_gray(), cfg=cfg)
        assert d.n_bands == 6

    def test_negative_fragment_id_raises(self):
        with pytest.raises(ValueError):
            extract_freq_descriptor(_gray(), fragment_id=-1)

    def test_default_config(self):
        d = extract_freq_descriptor(_gray())
        assert d.n_bands == 8

    def test_rgb_image(self):
        d = extract_freq_descriptor(_rgb())
        assert isinstance(d, FreqDescriptor)

    def test_centroid_in_0_1(self):
        d = extract_freq_descriptor(_gray())
        assert 0.0 <= d.centroid <= 1.0

    def test_entropy_nonnegative(self):
        d = extract_freq_descriptor(_gray())
        assert d.entropy >= 0.0


# ─── TestCompareFreqDescriptors ───────────────────────────────────────────────

class TestCompareFreqDescriptors:
    def _desc(self, bands):
        return FreqDescriptor(fragment_id=0, band_energies=bands,
                              centroid=0.3, top_freqs=[0.1], entropy=1.0)

    def test_identical_returns_one(self):
        bands = [1.0, 2.0, 3.0, 4.0]
        d = self._desc(bands)
        sim = compare_freq_descriptors(d, d)
        assert abs(sim - 1.0) < 1e-9

    def test_orthogonal_returns_zero(self):
        a = self._desc([1.0, 0.0, 0.0, 0.0])
        b = self._desc([0.0, 1.0, 0.0, 0.0])
        sim = compare_freq_descriptors(a, b)
        assert abs(sim - 0.0) < 1e-9

    def test_result_in_0_1(self):
        cfg = FreqConfig(n_bands=8)
        d1 = extract_freq_descriptor(_gray(seed=0), cfg=cfg)
        d2 = extract_freq_descriptor(_gray(seed=1), cfg=cfg)
        sim = compare_freq_descriptors(d1, d2)
        assert 0.0 <= sim <= 1.0

    def test_n_bands_mismatch_raises(self):
        a = self._desc([1.0, 2.0])
        b = self._desc([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            compare_freq_descriptors(a, b)


# ─── TestBatchExtractFreqDescriptors ─────────────────────────────────────────

class TestBatchExtractFreqDescriptors:
    def test_returns_list(self):
        r = batch_extract_freq_descriptors([_gray(), _rgb()])
        assert isinstance(r, list)

    def test_length_matches(self):
        imgs = [_gray(seed=i) for i in range(4)]
        r = batch_extract_freq_descriptors(imgs)
        assert len(r) == 4

    def test_each_is_freq_descriptor(self):
        r = batch_extract_freq_descriptors([_gray()])
        assert all(isinstance(d, FreqDescriptor) for d in r)

    def test_fragment_ids_sequential(self):
        imgs = [_gray(seed=i) for i in range(3)]
        r = batch_extract_freq_descriptors(imgs)
        assert [d.fragment_id for d in r] == [0, 1, 2]

    def test_empty_list(self):
        assert batch_extract_freq_descriptors([]) == []

    def test_custom_config(self):
        cfg = FreqConfig(n_bands=4)
        r = batch_extract_freq_descriptors([_gray()], cfg)
        assert r[0].n_bands == 4
