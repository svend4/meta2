"""
Tests for the new /api/v1/* endpoints added to tools/server.py.

Covers:
    GET  /api/v1/health
    GET  /api/v1/algorithms
    POST /api/v1/threshold
    GET  /api/v1/config/default
    POST /api/v1/config/validate
    GET  /api/v1/status
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.server import app as flask_app


# ─── Fixture ──────────────────────────────────────────────────────────────

@pytest.fixture()
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


# ─── Helper ───────────────────────────────────────────────────────────────

def _json(response):
    """Return the parsed JSON body of a response."""
    return response.get_json()


# ═══════════════════════════════════════════════════════════════════════════
# GET /api/v1/health
# ═══════════════════════════════════════════════════════════════════════════

class TestHealth:
    def test_health_returns_200(self, client):
        r = client.get("/api/v1/health")
        assert r.status_code == 200

    def test_health_response_has_status_key(self, client):
        data = _json(client.get("/api/v1/health"))
        assert "status" in data

    def test_health_status_is_ok(self, client):
        data = _json(client.get("/api/v1/health"))
        assert data["status"] == "ok"

    def test_health_response_has_version_key(self, client):
        data = _json(client.get("/api/v1/health"))
        assert "version" in data

    def test_health_version_value(self, client):
        data = _json(client.get("/api/v1/health"))
        assert data["version"] == "1.0"

    def test_health_response_has_algorithms_key(self, client):
        data = _json(client.get("/api/v1/health"))
        assert "algorithms" in data

    def test_health_algorithms_is_list_or_dict(self, client):
        data = _json(client.get("/api/v1/health"))
        assert isinstance(data["algorithms"], (list, dict))

    def test_health_algorithms_non_empty(self, client):
        data = _json(client.get("/api/v1/health"))
        assert len(data["algorithms"]) > 0

    def test_health_content_type_is_json(self, client):
        r = client.get("/api/v1/health")
        assert "application/json" in r.content_type


# ═══════════════════════════════════════════════════════════════════════════
# GET /api/v1/algorithms
# ═══════════════════════════════════════════════════════════════════════════

class TestAlgorithms:
    def test_algorithms_returns_200(self, client):
        r = client.get("/api/v1/algorithms")
        assert r.status_code == 200

    def test_algorithms_response_has_assembly_key(self, client):
        data = _json(client.get("/api/v1/algorithms"))
        assert "assembly" in data

    def test_algorithms_assembly_list_non_empty(self, client):
        data = _json(client.get("/api/v1/algorithms"))
        assert isinstance(data["assembly"], list)
        assert len(data["assembly"]) > 0

    def test_algorithms_response_has_matching_key(self, client):
        data = _json(client.get("/api/v1/algorithms"))
        assert "matching" in data

    def test_algorithms_matching_list_non_empty(self, client):
        data = _json(client.get("/api/v1/algorithms"))
        assert isinstance(data["matching"], list)
        assert len(data["matching"]) > 0

    def test_algorithms_response_has_preprocessing_key(self, client):
        data = _json(client.get("/api/v1/algorithms"))
        assert "preprocessing" in data

    def test_algorithms_preprocessing_list_non_empty(self, client):
        data = _json(client.get("/api/v1/algorithms"))
        assert isinstance(data["preprocessing"], list)
        assert len(data["preprocessing"]) > 0

    def test_algorithms_assembly_contains_beam(self, client):
        data = _json(client.get("/api/v1/algorithms"))
        assert "beam" in data["assembly"]

    def test_algorithms_assembly_contains_greedy(self, client):
        data = _json(client.get("/api/v1/algorithms"))
        assert "greedy" in data["assembly"]

    def test_algorithms_content_type_is_json(self, client):
        r = client.get("/api/v1/algorithms")
        assert "application/json" in r.content_type


# ═══════════════════════════════════════════════════════════════════════════
# GET /api/v1/config/default
# ═══════════════════════════════════════════════════════════════════════════

class TestConfigDefault:
    def test_config_default_returns_200(self, client):
        r = client.get("/api/v1/config/default")
        assert r.status_code == 200

    def test_config_default_response_is_dict(self, client):
        data = _json(client.get("/api/v1/config/default"))
        assert isinstance(data, dict)

    def test_config_default_has_matching_key(self, client):
        data = _json(client.get("/api/v1/config/default"))
        assert "matching" in data

    def test_config_default_has_assembly_key(self, client):
        data = _json(client.get("/api/v1/config/default"))
        assert "assembly" in data

    def test_config_default_has_preprocessing_key(self, client):
        data = _json(client.get("/api/v1/config/default"))
        assert "preprocessing" in data

    def test_config_default_has_segmentation_key(self, client):
        data = _json(client.get("/api/v1/config/default"))
        assert "segmentation" in data

    def test_config_default_assembly_method_present(self, client):
        data = _json(client.get("/api/v1/config/default"))
        assert "method" in data["assembly"]

    def test_config_default_content_type_is_json(self, client):
        r = client.get("/api/v1/config/default")
        assert "application/json" in r.content_type


# ═══════════════════════════════════════════════════════════════════════════
# POST /api/v1/config/validate
# ═══════════════════════════════════════════════════════════════════════════

class TestConfigValidate:
    def _post(self, client, body):
        return client.post(
            "/api/v1/config/validate",
            json=body,
            content_type="application/json",
        )

    def test_valid_config_returns_200(self, client):
        from puzzle_reconstruction.config import Config
        r = self._post(client, Config.default().to_dict())
        assert r.status_code == 200

    def test_valid_config_response_has_valid_key(self, client):
        from puzzle_reconstruction.config import Config
        data = _json(self._post(client, Config.default().to_dict()))
        assert "valid" in data

    def test_valid_config_returns_valid_true(self, client):
        from puzzle_reconstruction.config import Config
        data = _json(self._post(client, Config.default().to_dict()))
        assert data["valid"] is True

    def test_valid_config_errors_list_empty(self, client):
        from puzzle_reconstruction.config import Config
        data = _json(self._post(client, Config.default().to_dict()))
        assert data["errors"] == []

    def test_empty_dict_returns_200(self, client):
        r = self._post(client, {})
        assert r.status_code == 200

    def test_empty_dict_has_valid_key(self, client):
        data = _json(self._post(client, {}))
        assert "valid" in data

    def test_empty_dict_still_valid(self, client):
        # An empty dict should produce a default Config without errors
        data = _json(self._post(client, {}))
        assert data["valid"] is True

    def test_invalid_assembly_method_returns_errors(self, client):
        data = _json(self._post(client, {"assembly": {"method": "nonexistent_method"}}))
        assert data["valid"] is False
        assert len(data["errors"]) > 0

    def test_invalid_matching_threshold_too_large(self, client):
        data = _json(self._post(client, {"matching": {"threshold": 1.5}}))
        assert data["valid"] is False

    def test_invalid_section_type_returns_errors(self, client):
        # Passing a list instead of a dict for a section should flag an error
        data = _json(self._post(client, {"assembly": [1, 2, 3]}))
        assert data["valid"] is False
        assert len(data["errors"]) > 0

    def test_response_has_errors_key(self, client):
        data = _json(self._post(client, {}))
        assert "errors" in data

    def test_valid_partial_assembly_override(self, client):
        data = _json(self._post(client, {"assembly": {"method": "greedy"}}))
        assert data["valid"] is True


# ═══════════════════════════════════════════════════════════════════════════
# POST /api/v1/threshold
# ═══════════════════════════════════════════════════════════════════════════

_SAMPLE_MATRIX = [[0.1, 0.4, 0.7], [0.2, 0.5, 0.9], [0.0, 0.3, 0.6]]


class TestThreshold:
    def _post(self, client, body):
        return client.post(
            "/api/v1/threshold",
            json=body,
            content_type="application/json",
        )

    def test_threshold_returns_200(self, client):
        r = self._post(client, {"matrix": _SAMPLE_MATRIX})
        assert r.status_code == 200

    def test_threshold_response_has_threshold_key(self, client):
        data = _json(self._post(client, {"matrix": _SAMPLE_MATRIX}))
        assert "threshold" in data

    def test_threshold_value_in_range_0_1(self, client):
        data = _json(self._post(client, {"matrix": _SAMPLE_MATRIX}))
        assert 0.0 <= data["threshold"] <= 1.0

    def test_threshold_response_has_method_key(self, client):
        data = _json(self._post(client, {"matrix": _SAMPLE_MATRIX}))
        assert "method" in data

    def test_threshold_method_otsu_works(self, client):
        data = _json(self._post(client, {"matrix": _SAMPLE_MATRIX, "method": "otsu"}))
        assert data["method"] == "otsu"
        assert 0.0 <= data["threshold"] <= 1.0

    def test_threshold_method_percentile_works(self, client):
        data = _json(self._post(client, {
            "matrix": _SAMPLE_MATRIX,
            "method": "percentile",
            "percentile": 75,
        }))
        assert data["method"] == "percentile"
        assert 0.0 <= data["threshold"] <= 1.0

    def test_threshold_percentile_50_is_median(self, client):
        import numpy as np
        flat = [v for row in _SAMPLE_MATRIX for v in row]
        expected = float(np.percentile(flat, 50))
        data = _json(self._post(client, {
            "matrix": _SAMPLE_MATRIX,
            "method": "percentile",
            "percentile": 50,
        }))
        assert abs(data["threshold"] - expected) < 1e-4

    def test_threshold_missing_matrix_returns_400(self, client):
        r = self._post(client, {"method": "otsu"})
        assert r.status_code == 400

    def test_threshold_unknown_method_returns_400(self, client):
        r = self._post(client, {"matrix": _SAMPLE_MATRIX, "method": "unknown_algo"})
        assert r.status_code == 400

    def test_threshold_default_method_is_otsu(self, client):
        data = _json(self._post(client, {"matrix": _SAMPLE_MATRIX}))
        assert data["method"] == "otsu"

    def test_threshold_content_type_is_json(self, client):
        r = self._post(client, {"matrix": _SAMPLE_MATRIX})
        assert "application/json" in r.content_type

    def test_threshold_1x1_matrix(self, client):
        r = self._post(client, {"matrix": [[0.5]], "method": "percentile", "percentile": 50})
        assert r.status_code == 200
        data = _json(r)
        assert data["threshold"] == pytest.approx(0.5, abs=1e-4)


# ═══════════════════════════════════════════════════════════════════════════
# GET /api/v1/status
# ═══════════════════════════════════════════════════════════════════════════

class TestStatus:
    def test_status_returns_200(self, client):
        r = client.get("/api/v1/status")
        assert r.status_code == 200

    def test_status_has_available_methods_key(self, client):
        data = _json(client.get("/api/v1/status"))
        assert "available_methods" in data

    def test_status_has_n_fragments_supported_key(self, client):
        data = _json(client.get("/api/v1/status"))
        assert "n_fragments_supported" in data

    def test_status_n_fragments_is_list(self, client):
        data = _json(client.get("/api/v1/status"))
        assert isinstance(data["n_fragments_supported"], list)

    def test_status_n_fragments_values(self, client):
        data = _json(client.get("/api/v1/status"))
        for n in [4, 9, 16, 25]:
            assert n in data["n_fragments_supported"]

    def test_status_available_methods_is_list(self, client):
        data = _json(client.get("/api/v1/status"))
        assert isinstance(data["available_methods"], list)

    def test_status_available_methods_non_empty(self, client):
        data = _json(client.get("/api/v1/status"))
        assert len(data["available_methods"]) > 0

    def test_status_content_type_is_json(self, client):
        r = client.get("/api/v1/status")
        assert "application/json" in r.content_type


# ═══════════════════════════════════════════════════════════════════════════
# Routing / 404
# ═══════════════════════════════════════════════════════════════════════════

class TestRouting:
    def test_invalid_endpoint_returns_404(self, client):
        r = client.get("/api/v1/does_not_exist")
        assert r.status_code == 404

    def test_misspelled_health_returns_404(self, client):
        r = client.get("/api/v1/healthz")
        assert r.status_code == 404

    def test_wrong_method_on_health_returns_405(self, client):
        r = client.post("/api/v1/health")
        assert r.status_code == 405

    def test_wrong_method_on_status_returns_405(self, client):
        r = client.post("/api/v1/status")
        assert r.status_code == 405
