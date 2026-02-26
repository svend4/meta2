"""
Integration tests for tools/server.py Flask REST API.

Uses the Flask test client (app.test_client()) — no mocks for HTTP endpoints.
"""

import sys
import json
import pytest

sys.path.insert(0, '/home/user/meta2')

from tools.server import app, _float_or_none, _int_or_none


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Return a Flask test client for the entire module."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ===========================================================================
# TestHealthEndpoint
# ===========================================================================

class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_status_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_returns_json(self, client):
        resp = client.get("/health")
        data = resp.get_json()
        assert data is not None, "Response must be valid JSON"

    def test_health_has_required_keys(self, client):
        data = client.get("/health").get_json()
        for key in ("status", "version", "jobs", "uptime_s", "timestamp"):
            assert key in data, f"Key '{key}' missing from /health response"

    def test_health_status_is_ok(self, client):
        data = client.get("/health").get_json()
        assert data["status"] == "ok"

    def test_health_version_is_1_0_0(self, client):
        data = client.get("/health").get_json()
        assert data["version"] == "1.0.0"

    def test_health_jobs_is_non_negative_integer(self, client):
        data = client.get("/health").get_json()
        assert isinstance(data["jobs"], int)
        assert data["jobs"] >= 0

    def test_health_uptime_is_non_negative(self, client):
        data = client.get("/health").get_json()
        assert data["uptime_s"] >= 0


# ===========================================================================
# TestSpecEndpoint
# ===========================================================================

class TestSpecEndpoint:
    """Tests for GET /spec (OpenAPI 3.0)."""

    def test_spec_status_200(self, client):
        resp = client.get("/spec")
        assert resp.status_code == 200

    def test_spec_openapi_version(self, client):
        data = client.get("/spec").get_json()
        assert data.get("openapi") == "3.0.3"

    def test_spec_has_info_with_title(self, client):
        data = client.get("/spec").get_json()
        assert "info" in data
        assert "title" in data["info"]

    def test_spec_has_paths_dict(self, client):
        data = client.get("/spec").get_json()
        assert "paths" in data
        assert isinstance(data["paths"], dict)

    def test_spec_paths_contains_health(self, client):
        paths = client.get("/spec").get_json()["paths"]
        assert "/health" in paths

    def test_spec_paths_contains_api_methods(self, client):
        paths = client.get("/spec").get_json()["paths"]
        assert "/api/methods" in paths

    def test_spec_paths_contains_api_reconstruct(self, client):
        paths = client.get("/spec").get_json()["paths"]
        assert "/api/reconstruct" in paths


# ===========================================================================
# TestMethodsEndpoint
# ===========================================================================

class TestMethodsEndpoint:
    """Tests for GET /api/methods."""

    def test_methods_status_200(self, client):
        resp = client.get("/api/methods")
        assert resp.status_code == 200

    def test_methods_has_methods_list(self, client):
        data = client.get("/api/methods").get_json()
        assert "methods" in data
        assert isinstance(data["methods"], list)

    def test_methods_list_length_is_10(self, client):
        data = client.get("/api/methods").get_json()
        assert len(data["methods"]) == 10

    def test_each_method_has_id_and_desc(self, client):
        methods = client.get("/api/methods").get_json()["methods"]
        for m in methods:
            assert "id" in m, f"Method entry missing 'id': {m}"
            assert "desc" in m, f"Method entry missing 'desc': {m}"

    def test_default_key_is_beam(self, client):
        data = client.get("/api/methods").get_json()
        assert data.get("default") == "beam"

    def test_method_ids_include_greedy(self, client):
        ids = [m["id"] for m in client.get("/api/methods").get_json()["methods"]]
        assert "greedy" in ids

    def test_method_ids_include_beam(self, client):
        ids = [m["id"] for m in client.get("/api/methods").get_json()["methods"]]
        assert "beam" in ids

    def test_method_ids_include_sa(self, client):
        ids = [m["id"] for m in client.get("/api/methods").get_json()["methods"]]
        assert "sa" in ids

    def test_method_ids_include_gamma(self, client):
        ids = [m["id"] for m in client.get("/api/methods").get_json()["methods"]]
        assert "gamma" in ids

    def test_method_ids_include_genetic(self, client):
        ids = [m["id"] for m in client.get("/api/methods").get_json()["methods"]]
        assert "genetic" in ids


# ===========================================================================
# TestConfigEndpoint
# ===========================================================================

class TestConfigEndpoint:
    """Tests for GET /config."""

    def test_config_status_200(self, client):
        resp = client.get("/config")
        assert resp.status_code == 200

    def test_config_returns_dict(self, client):
        data = client.get("/config").get_json()
        assert isinstance(data, dict)

    def test_config_has_assembly_section(self, client):
        data = client.get("/config").get_json()
        assert "assembly" in data

    def test_config_has_matching_section(self, client):
        data = client.get("/config").get_json()
        assert "matching" in data

    def test_config_has_preprocessing_section(self, client):
        data = client.get("/config").get_json()
        assert "preprocessing" in data

    def test_config_has_verification_section(self, client):
        data = client.get("/config").get_json()
        assert "verification" in data


# ===========================================================================
# TestV1HealthAlgorithms
# ===========================================================================

class TestV1HealthAlgorithms:
    """Tests for GET /api/v1/health and GET /api/v1/algorithms."""

    def test_v1_health_status_200(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_v1_health_status_is_ok(self, client):
        data = client.get("/api/v1/health").get_json()
        assert data.get("status") == "ok"

    def test_v1_health_has_version(self, client):
        data = client.get("/api/v1/health").get_json()
        assert "version" in data

    def test_v1_health_has_algorithms_list(self, client):
        data = client.get("/api/v1/health").get_json()
        assert "algorithms" in data
        assert isinstance(data["algorithms"], list)

    def test_v1_algorithms_status_200(self, client):
        resp = client.get("/api/v1/algorithms")
        assert resp.status_code == 200

    def test_v1_algorithms_has_assembly_key(self, client):
        data = client.get("/api/v1/algorithms").get_json()
        assert "assembly" in data
        assert isinstance(data["assembly"], list)

    def test_v1_algorithms_has_matching_key(self, client):
        data = client.get("/api/v1/algorithms").get_json()
        assert "matching" in data
        assert isinstance(data["matching"], list)

    def test_v1_algorithms_has_preprocessing_key(self, client):
        data = client.get("/api/v1/algorithms").get_json()
        assert "preprocessing" in data
        assert isinstance(data["preprocessing"], list)

    def test_v1_algorithms_assembly_includes_greedy(self, client):
        data = client.get("/api/v1/algorithms").get_json()
        assert "greedy" in data["assembly"]

    def test_v1_algorithms_assembly_includes_beam(self, client):
        data = client.get("/api/v1/algorithms").get_json()
        assert "beam" in data["assembly"]


# ===========================================================================
# TestV1ThresholdEndpoint
# ===========================================================================

class TestV1ThresholdEndpoint:
    """Tests for POST /api/v1/threshold."""

    def test_threshold_default_method_200(self, client):
        resp = client.post(
            "/api/v1/threshold",
            json={"matrix": [[0.1, 0.5], [0.8, 0.3]]},
        )
        assert resp.status_code == 200

    def test_threshold_otsu_method_200(self, client):
        resp = client.post(
            "/api/v1/threshold",
            json={"matrix": [[0.1, 0.5]], "method": "otsu"},
        )
        assert resp.status_code == 200

    def test_threshold_otsu_value_in_0_1(self, client):
        data = client.post(
            "/api/v1/threshold",
            json={"matrix": [[0.1, 0.5]], "method": "otsu"},
        ).get_json()
        assert "threshold" in data
        assert 0.0 <= data["threshold"] <= 1.0

    def test_threshold_percentile_200(self, client):
        resp = client.post(
            "/api/v1/threshold",
            json={"matrix": [[0.1, 0.5]], "method": "percentile", "percentile": 50},
        )
        assert resp.status_code == 200

    def test_threshold_percentile_value_in_response(self, client):
        data = client.post(
            "/api/v1/threshold",
            json={"matrix": [[0.1, 0.5]], "method": "percentile", "percentile": 50},
        ).get_json()
        assert "threshold" in data

    def test_threshold_missing_matrix_400(self, client):
        resp = client.post("/api/v1/threshold", json={})
        assert resp.status_code == 400

    def test_threshold_empty_matrix_400(self, client):
        resp = client.post("/api/v1/threshold", json={"matrix": []})
        assert resp.status_code == 400

    def test_threshold_unknown_method_400(self, client):
        resp = client.post(
            "/api/v1/threshold",
            json={"matrix": [[0.5]], "method": "unknown_method"},
        )
        assert resp.status_code == 400

    def test_threshold_result_is_float(self, client):
        data = client.post(
            "/api/v1/threshold",
            json={"matrix": [[0.1, 0.5], [0.8, 0.3]]},
        ).get_json()
        assert isinstance(data["threshold"], float)

    def test_threshold_method_field_in_response(self, client):
        data = client.post(
            "/api/v1/threshold",
            json={"matrix": [[0.1, 0.5]], "method": "otsu"},
        ).get_json()
        assert data.get("method") == "otsu"


# ===========================================================================
# TestV1ConfigValidate
# ===========================================================================

class TestV1ConfigValidate:
    """Tests for POST /api/v1/config/validate."""

    def test_validate_empty_body_200(self, client):
        resp = client.post("/api/v1/config/validate", json={})
        assert resp.status_code == 200

    def test_validate_empty_body_valid_true(self, client):
        data = client.post("/api/v1/config/validate", json={}).get_json()
        assert data["valid"] is True
        assert data["errors"] == []

    def test_validate_beam_method_valid(self, client):
        data = client.post(
            "/api/v1/config/validate",
            json={"assembly": {"method": "beam"}},
        ).get_json()
        assert data["valid"] is True

    def test_validate_invalid_method_not_valid(self, client):
        data = client.post(
            "/api/v1/config/validate",
            json={"assembly": {"method": "invalid_method"}},
        ).get_json()
        assert data["valid"] is False
        assert len(data["errors"]) > 0

    def test_validate_matching_threshold_ok(self, client):
        data = client.post(
            "/api/v1/config/validate",
            json={"matching": {"threshold": 0.5}},
        ).get_json()
        assert data["valid"] is True

    def test_validate_matching_threshold_out_of_range(self, client):
        data = client.post(
            "/api/v1/config/validate",
            json={"matching": {"threshold": 1.5}},
        ).get_json()
        assert data["valid"] is False

    def test_validate_assembly_not_dict(self, client):
        data = client.post(
            "/api/v1/config/validate",
            json={"assembly": "not_a_dict"},
        ).get_json()
        assert data["valid"] is False

    def test_validate_response_has_valid_key(self, client):
        data = client.post("/api/v1/config/validate", json={}).get_json()
        assert "valid" in data

    def test_validate_response_has_errors_key(self, client):
        data = client.post("/api/v1/config/validate", json={}).get_json()
        assert "errors" in data
        assert isinstance(data["errors"], list)


# ===========================================================================
# TestV1StatusDefault
# ===========================================================================

class TestV1StatusDefault:
    """Tests for GET /api/v1/status and GET /api/v1/config/default."""

    def test_v1_status_200(self, client):
        resp = client.get("/api/v1/status")
        assert resp.status_code == 200

    def test_v1_status_has_n_fragments_supported(self, client):
        data = client.get("/api/v1/status").get_json()
        assert "n_fragments_supported" in data
        assert isinstance(data["n_fragments_supported"], list)

    def test_v1_status_has_available_methods(self, client):
        data = client.get("/api/v1/status").get_json()
        assert "available_methods" in data
        assert isinstance(data["available_methods"], list)

    def test_v1_status_n_fragments_includes_4(self, client):
        data = client.get("/api/v1/status").get_json()
        assert 4 in data["n_fragments_supported"]

    def test_v1_status_n_fragments_includes_9(self, client):
        data = client.get("/api/v1/status").get_json()
        assert 9 in data["n_fragments_supported"]

    def test_v1_status_n_fragments_includes_16(self, client):
        data = client.get("/api/v1/status").get_json()
        assert 16 in data["n_fragments_supported"]

    def test_v1_config_default_200(self, client):
        resp = client.get("/api/v1/config/default")
        assert resp.status_code == 200

    def test_v1_config_default_returns_dict(self, client):
        data = client.get("/api/v1/config/default").get_json()
        assert isinstance(data, dict)


# ===========================================================================
# TestErrorCases
# ===========================================================================

class TestErrorCases:
    """Tests for error responses from various endpoints."""

    def test_reconstruct_no_files_400(self, client):
        resp = client.post("/api/reconstruct")
        assert resp.status_code == 400

    def test_reconstruct_no_files_has_error_message(self, client):
        data = client.post("/api/reconstruct").get_json()
        assert "error" in data

    def test_cluster_no_files_400(self, client):
        resp = client.post("/api/cluster")
        assert resp.status_code == 400

    def test_cluster_no_files_has_error_message(self, client):
        data = client.post("/api/cluster").get_json()
        assert "error" in data

    def test_report_unknown_job_id_404(self, client):
        resp = client.get("/api/report/nonexistent-job-id-12345")
        assert resp.status_code == 404

    def test_report_unknown_job_id_has_error(self, client):
        data = client.get("/api/report/nonexistent-job-id-12345").get_json()
        assert "error" in data

    def test_threshold_no_matrix_error_message(self, client):
        data = client.post("/api/v1/threshold", json={}).get_json()
        assert "error" in data

    def test_threshold_unknown_method_error_message(self, client):
        data = client.post(
            "/api/v1/threshold",
            json={"matrix": [[0.5]], "method": "bad"},
        ).get_json()
        assert "error" in data


# ===========================================================================
# TestHelperFunctions
# ===========================================================================

class TestHelperFunctions:
    """Tests for the _float_or_none and _int_or_none helper functions."""

    def test_float_or_none_with_none(self):
        assert _float_or_none(None) is None

    def test_float_or_none_with_valid_string(self):
        assert _float_or_none("0.5") == 0.5

    def test_float_or_none_with_invalid_string(self):
        assert _float_or_none("bad") is None

    def test_float_or_none_with_integer_string(self):
        result = _float_or_none("1")
        assert result == 1.0
        assert isinstance(result, float)

    def test_float_or_none_with_negative_string(self):
        assert _float_or_none("-3.14") == pytest.approx(-3.14)

    def test_int_or_none_with_none(self):
        assert _int_or_none(None) is None

    def test_int_or_none_with_valid_string(self):
        assert _int_or_none("3") == 3

    def test_int_or_none_with_invalid_string(self):
        assert _int_or_none("bad") is None

    def test_int_or_none_with_zero_string(self):
        assert _int_or_none("0") == 0

    def test_int_or_none_with_negative_string(self):
        assert _int_or_none("-5") == -5
