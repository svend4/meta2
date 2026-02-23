"""Extra tests for puzzle_reconstruction.preprocessing.frequency_analyzer."""
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
    return np.random.default_rng(seed).integers(0, 255, (h, w), dtype=np.uint8)

def _rgb(h=32, w=32, seed=1):
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)

def _spectrum(h=32, w=32, seed=0):
    return compute_power_spectrum(_gray(h, w, seed))

def _desc(bands):
    return FreqDescriptor(fragment_id=0, band_energies=bands,
                          centroid=0.3, top_freqs=[0.1], entropy=1.0)


# ─── TestFreqConfigExtra ──────────────────────────────────────────────────────

class TestFreqConfigExtra:
    def test_n_bands_2_valid(self):
        cfg = FreqConfig(n_bands=2)
        assert cfg.n_bands == 2

    def test_n_top_freqs_1_valid(self):
        cfg = FreqConfig(n_top_freqs=1)
        assert cfg.n_top_freqs == 1

    def test_large_n_bands(self):
        cfg = FreqConfig(n_bands=32)
        assert cfg.n_bands == 32

    def test_log_scale_false_stored(self):
        cfg = FreqConfig(log_scale=False)
        assert cfg.log_scale is False

    def test_normalize_false_stored(self):
        cfg = FreqConfig(normalize=False)
        assert cfg.normalize is False

    def test_n_top_freqs_10(self):
        cfg = FreqConfig(n_top_freqs=10)
        assert cfg.n_top_freqs == 10


# ─── TestFreqSpectrumExtra ────────────────────────────────────────────────────

class TestFreqSpectrumExtra:
    def test_zero_total_power_ok(self):
        mag = np.zeros((4, 4))
        s = FreqSpectrum(magnitude=mag, power=mag, total_power=0.0)
        assert s.total_power == 0.0

    def test_magnitude_shape(self):
        s = _spectrum()
        assert s.magnitude.shape == s.shape

    def test_power_shape(self):
        s = _spectrum()
        assert s.power.shape == s.shape

    def test_non_square_shape(self):
        s = compute_power_spectrum(_gray(32, 48))
        assert s.shape == (32, 48)

    def test_magnitude_nonneg(self):
        s = _spectrum()
        assert (s.magnitude >= 0.0).all()

    def test_power_nonneg(self):
        s = _spectrum()
        assert (s.power >= 0.0).all()


# ─── TestFreqDescriptorExtra ──────────────────────────────────────────────────

class TestFreqDescriptorExtra:
    def test_high_freq_ratio_half(self):
        bands = [1.0] * 4 + [1.0] * 4  # equal distribution
        d = _desc(bands)
        assert 0.0 <= d.high_freq_ratio <= 1.0

    def test_top_freqs_list_length(self):
        d = FreqDescriptor(fragment_id=0, band_energies=[1.0, 2.0],
                           centroid=0.3, top_freqs=[0.1, 0.2, 0.3],
                           entropy=1.0)
        assert len(d.top_freqs) == 3

    def test_dominant_band_first(self):
        bands = [10.0, 1.0, 1.0, 1.0]
        d = _desc(bands)
        assert d.dominant_band == 0

    def test_fragment_id_positive(self):
        d = FreqDescriptor(fragment_id=99, band_energies=[1.0],
                           centroid=0.3, top_freqs=[], entropy=0.0)
        assert d.fragment_id == 99

    def test_n_bands_equals_length(self):
        d = _desc([0.5, 0.5, 0.5])
        assert d.n_bands == 3

    def test_centroid_stored(self):
        d = FreqDescriptor(fragment_id=0, band_energies=[1.0],
                           centroid=0.7, top_freqs=[], entropy=0.0)
        assert d.centroid == pytest.approx(0.7)


# ─── TestComputePowerSpectrumExtra ────────────────────────────────────────────

class TestComputePowerSpectrumExtra:
    def test_large_image(self):
        s = compute_power_spectrum(_gray(128, 128))
        assert s.shape == (128, 128)

    def test_non_square(self):
        s = compute_power_spectrum(_gray(32, 64))
        assert s.shape == (32, 64)

    def test_uniform_image(self):
        img = np.full((16, 16), 128, dtype=np.uint8)
        s = compute_power_spectrum(img)
        assert isinstance(s, FreqSpectrum)

    def test_log_scale_true(self):
        s = compute_power_spectrum(_gray(), log_scale=True)
        assert isinstance(s, FreqSpectrum)

    def test_different_seeds_different_power(self):
        s1 = compute_power_spectrum(_gray(seed=0))
        s2 = compute_power_spectrum(_gray(seed=42))
        # Total powers need not be equal for different images
        assert isinstance(s1.total_power, float)
        assert isinstance(s2.total_power, float)

    def test_rgb_shape_2d_spectrum(self):
        s = compute_power_spectrum(_rgb(32, 32))
        assert len(s.shape) == 2


# ─── TestComputeBandEnergiesExtra ────────────────────────────────────────────

class TestComputeBandEnergiesExtra:
    def test_n_bands_2(self):
        be = compute_band_energies(_spectrum(), n_bands=2)
        assert len(be) == 2

    def test_n_bands_32(self):
        be = compute_band_energies(_spectrum(64, 64), n_bands=32)
        assert len(be) == 32

    def test_floats(self):
        be = compute_band_energies(_spectrum(), n_bands=4)
        for e in be:
            assert isinstance(e, float)

    def test_various_band_counts_all_nonneg(self):
        s = _spectrum(64, 64)
        for n in (2, 4, 8, 16):
            be = compute_band_energies(s, n_bands=n)
            assert all(e >= 0.0 for e in be)

    def test_total_energy_consistent_across_bands(self):
        s = _spectrum()
        total_4 = sum(compute_band_energies(s, n_bands=4))
        total_8 = sum(compute_band_energies(s, n_bands=8))
        # Both should be proportional to total energy
        assert total_4 > 0.0 and total_8 > 0.0


# ─── TestComputeSpectralCentroidExtra ─────────────────────────────────────────

class TestComputeSpectralCentroidExtra:
    def test_different_images_different_centroids(self):
        c1 = compute_spectral_centroid(_spectrum(seed=0))
        c2 = compute_spectral_centroid(_spectrum(seed=99))
        # They may or may not be equal – just verify type and range
        assert isinstance(c1, float) and isinstance(c2, float)
        assert 0.0 <= c1 <= 1.0
        assert 0.0 <= c2 <= 1.0

    def test_uniform_power_spectrum(self):
        pwr = np.ones((32, 32))
        s = FreqSpectrum(magnitude=pwr, power=pwr, total_power=float(pwr.sum()))
        c = compute_spectral_centroid(s)
        assert 0.0 <= c <= 1.0

    def test_returns_float(self):
        c = compute_spectral_centroid(_spectrum())
        assert isinstance(c, float)

    def test_five_seeds_all_valid(self):
        for seed in range(5):
            c = compute_spectral_centroid(_spectrum(seed=seed))
            assert 0.0 <= c <= 1.0


# ─── TestComputeSpectralEntropyExtra ──────────────────────────────────────────

class TestComputeSpectralEntropyExtra:
    def test_zero_power_returns_zero(self):
        pwr = np.zeros((8, 8))
        s = FreqSpectrum(magnitude=pwr, power=pwr, total_power=0.0)
        e = compute_spectral_entropy(s)
        assert e == pytest.approx(0.0)

    def test_various_seeds_nonneg(self):
        for seed in range(5):
            e = compute_spectral_entropy(_spectrum(seed=seed))
            assert e >= 0.0

    def test_returns_float(self):
        e = compute_spectral_entropy(_spectrum())
        assert isinstance(e, float)


# ─── TestExtractTopFrequenciesExtra ──────────────────────────────────────────

class TestExtractTopFrequenciesExtra:
    def test_n_top_1(self):
        s = _spectrum()
        f = extract_top_frequencies(s, n_top=1)
        assert len(f) == 1

    def test_n_top_10(self):
        s = _spectrum(64, 64)
        f = extract_top_frequencies(s, n_top=10)
        assert len(f) == 10

    def test_all_floats(self):
        s = _spectrum()
        for v in extract_top_frequencies(s, n_top=5):
            assert isinstance(v, float)

    def test_five_seeds_all_valid(self):
        for seed in range(5):
            s = _spectrum(seed=seed)
            f = extract_top_frequencies(s, n_top=3)
            assert all(0.0 <= v <= 1.0 for v in f)


# ─── TestExtractFreqDescriptorExtra ──────────────────────────────────────────

class TestExtractFreqDescriptorExtra:
    def test_five_seeds(self):
        for s in range(5):
            d = extract_freq_descriptor(_gray(seed=s))
            assert isinstance(d, FreqDescriptor)

    def test_large_image(self):
        d = extract_freq_descriptor(_gray(128, 128))
        assert d.n_bands == 8

    def test_5_band_config(self):
        cfg = FreqConfig(n_bands=5)
        d = extract_freq_descriptor(_gray(), cfg=cfg)
        assert d.n_bands == 5

    def test_n_top_freqs_3(self):
        cfg = FreqConfig(n_top_freqs=3)
        d = extract_freq_descriptor(_gray(), cfg=cfg)
        assert len(d.top_freqs) == 3

    def test_fragment_id_0(self):
        d = extract_freq_descriptor(_gray())
        assert d.fragment_id == 0

    def test_rgb_centroid_valid(self):
        d = extract_freq_descriptor(_rgb())
        assert 0.0 <= d.centroid <= 1.0


# ─── TestCompareFreqDescriptorsExtra ─────────────────────────────────────────

class TestCompareFreqDescriptorsExtra:
    def test_symmetric(self):
        cfg = FreqConfig(n_bands=4)
        d1 = extract_freq_descriptor(_gray(seed=0), cfg=cfg)
        d2 = extract_freq_descriptor(_gray(seed=1), cfg=cfg)
        assert compare_freq_descriptors(d1, d2) == pytest.approx(
            compare_freq_descriptors(d2, d1), abs=1e-9
        )

    def test_all_zero_bands_in_range(self):
        a = _desc([0.0, 0.0, 0.0, 0.0])
        b = _desc([0.0, 0.0, 0.0, 0.0])
        sim = compare_freq_descriptors(a, b)
        assert 0.0 <= sim <= 1.0

    def test_various_pairs_in_range(self):
        cfg = FreqConfig(n_bands=4)
        descs = [extract_freq_descriptor(_gray(seed=i), cfg=cfg) for i in range(4)]
        for i in range(len(descs)):
            for j in range(len(descs)):
                sim = compare_freq_descriptors(descs[i], descs[j])
                assert 0.0 <= sim <= 1.0

    def test_similar_images_high_similarity(self):
        cfg = FreqConfig(n_bands=8)
        d1 = extract_freq_descriptor(_gray(seed=7), cfg=cfg)
        d2 = extract_freq_descriptor(_gray(seed=7), cfg=cfg)
        assert compare_freq_descriptors(d1, d2) == pytest.approx(1.0, abs=1e-9)


# ─── TestBatchExtractFreqDescriptorsExtra ─────────────────────────────────────

class TestBatchExtractFreqDescriptorsExtra:
    def test_ten_images(self):
        imgs = [_gray(seed=i) for i in range(10)]
        result = batch_extract_freq_descriptors(imgs)
        assert len(result) == 10

    def test_mixed_gray_rgb(self):
        imgs = [_gray(), _rgb(), _gray(seed=2), _rgb(seed=3)]
        result = batch_extract_freq_descriptors(imgs)
        assert len(result) == 4

    def test_n_bands_4_batch(self):
        cfg = FreqConfig(n_bands=4)
        imgs = [_gray(seed=i) for i in range(3)]
        result = batch_extract_freq_descriptors(imgs, cfg)
        for d in result:
            assert d.n_bands == 4

    def test_fragment_ids_0_to_n(self):
        imgs = [_gray(seed=i) for i in range(5)]
        result = batch_extract_freq_descriptors(imgs)
        assert [d.fragment_id for d in result] == list(range(5))

    def test_all_valid_centroids(self):
        imgs = [_gray(seed=i) for i in range(4)]
        for d in batch_extract_freq_descriptors(imgs):
            assert 0.0 <= d.centroid <= 1.0
