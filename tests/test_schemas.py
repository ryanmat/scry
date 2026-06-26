# Description: Tests for API request/response Pydantic schemas.
# Description: Validates serialization, validation, and field constraints.

"""Tests for API schemas."""

import pytest
from pydantic import ValidationError


class TestPredictionRequest:
    """Tests for PredictionRequest schema."""

    def test_valid_request(self):
        """Test creating a valid prediction request."""
        from scry.api.schemas import PredictionRequest

        request = PredictionRequest(
            resource_id="test-pod-123",
            numerical_metrics={
                "cpuUsageNanoCores": [1000000.0, 1100000.0, 1200000.0],
                "memoryUsageBytes": [50000000.0, 51000000.0, 52000000.0],
            },
            categorical_metrics={
                "kubePodStatusReady": [1, 1, 1],
                "podConditionPhase": [1, 1, 1],
            },
        )

        assert request.resource_id == "test-pod-123"
        assert len(request.numerical_metrics) == 2
        assert len(request.categorical_metrics) == 2

    def test_default_window_minutes(self):
        """Test default window_minutes is 30."""
        from scry.api.schemas import PredictionRequest

        request = PredictionRequest(
            resource_id="test-pod",
            numerical_metrics={"cpu": [1.0]},
            categorical_metrics={"ready": [1]},
        )

        assert request.window_minutes == 30

    def test_custom_window_minutes(self):
        """Test custom window_minutes."""
        from scry.api.schemas import PredictionRequest

        request = PredictionRequest(
            resource_id="test-pod",
            numerical_metrics={"cpu": [1.0]},
            categorical_metrics={"ready": [1]},
            window_minutes=60,
        )

        assert request.window_minutes == 60

    def test_empty_resource_id_fails(self):
        """Test that empty resource_id is rejected."""
        from scry.api.schemas import PredictionRequest

        with pytest.raises(ValidationError):
            PredictionRequest(
                resource_id="",
                numerical_metrics={"cpu": [1.0]},
                categorical_metrics={"ready": [1]},
            )

    def test_empty_metrics_fails(self):
        """Test that empty metrics dicts are rejected."""
        from scry.api.schemas import PredictionRequest

        with pytest.raises(ValidationError):
            PredictionRequest(
                resource_id="test-pod",
                numerical_metrics={},
                categorical_metrics={"ready": [1]},
            )

    def test_serialization(self):
        """Test request serializes to dict correctly."""
        from scry.api.schemas import PredictionRequest

        request = PredictionRequest(
            resource_id="test-pod",
            numerical_metrics={"cpu": [1.0, 2.0]},
            categorical_metrics={"ready": [1, 1]},
        )

        data = request.model_dump()
        assert data["resource_id"] == "test-pod"
        assert data["numerical_metrics"]["cpu"] == [1.0, 2.0]


class TestPredictionResponse:
    """Tests for PredictionResponse schema."""

    def test_valid_response(self):
        """Test creating a valid prediction response."""
        from scry.api.schemas import PredictionResponse

        response = PredictionResponse(
            resource_id="test-pod-123",
            cluster_id=0,
            cluster_name="NORMAL",
            confidence=0.95,
            action="NONE",
            priority="LOW",
        )

        assert response.cluster_id == 0
        assert response.cluster_name == "NORMAL"
        assert response.confidence == 0.95
        assert response.action == "NONE"
        assert response.priority == "LOW"

    def test_all_cluster_names_valid(self):
        """Test all valid cluster names."""
        from scry.api.schemas import PredictionResponse

        valid_names = ["NORMAL", "PRE_SCALE", "PRE_FAILURE", "ACTIVE_DEGRADATION", "ANOMALY"]

        for i, name in enumerate(valid_names):
            response = PredictionResponse(
                resource_id="test",
                cluster_id=i,
                cluster_name=name,
                confidence=0.9,
                action="NONE",
                priority="LOW",
            )
            assert response.cluster_name == name

    def test_invalid_cluster_name_fails(self):
        """Test that invalid cluster name is rejected."""
        from scry.api.schemas import PredictionResponse

        with pytest.raises(ValidationError):
            PredictionResponse(
                resource_id="test",
                cluster_id=0,
                cluster_name="INVALID",
                confidence=0.9,
                action="NONE",
                priority="LOW",
            )

    def test_all_actions_valid(self):
        """Test all valid action values."""
        from scry.api.schemas import PredictionResponse

        valid_actions = ["NONE", "SCALE", "DIAGNOSTIC", "REMEDIATE", "ALERT"]

        for action in valid_actions:
            response = PredictionResponse(
                resource_id="test",
                cluster_id=0,
                cluster_name="NORMAL",
                confidence=0.9,
                action=action,
                priority="LOW",
            )
            assert response.action == action

    def test_invalid_action_fails(self):
        """Test that invalid action is rejected."""
        from scry.api.schemas import PredictionResponse

        with pytest.raises(ValidationError):
            PredictionResponse(
                resource_id="test",
                cluster_id=0,
                cluster_name="NORMAL",
                confidence=0.9,
                action="INVALID",
                priority="LOW",
            )

    def test_all_priorities_valid(self):
        """Test all valid priority values."""
        from scry.api.schemas import PredictionResponse

        valid_priorities = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

        for priority in valid_priorities:
            response = PredictionResponse(
                resource_id="test",
                cluster_id=0,
                cluster_name="NORMAL",
                confidence=0.9,
                action="NONE",
                priority=priority,
            )
            assert response.priority == priority

    def test_invalid_priority_fails(self):
        """Test that invalid priority is rejected."""
        from scry.api.schemas import PredictionResponse

        with pytest.raises(ValidationError):
            PredictionResponse(
                resource_id="test",
                cluster_id=0,
                cluster_name="NORMAL",
                confidence=0.9,
                action="NONE",
                priority="INVALID",
            )

    def test_confidence_range(self):
        """Test confidence must be between 0 and 1."""
        from scry.api.schemas import PredictionResponse

        # Valid boundary values
        PredictionResponse(
            resource_id="test",
            cluster_id=0,
            cluster_name="NORMAL",
            confidence=0.0,
            action="NONE",
            priority="LOW",
        )
        PredictionResponse(
            resource_id="test",
            cluster_id=0,
            cluster_name="NORMAL",
            confidence=1.0,
            action="NONE",
            priority="LOW",
        )

        # Invalid: > 1
        with pytest.raises(ValidationError):
            PredictionResponse(
                resource_id="test",
                cluster_id=0,
                cluster_name="NORMAL",
                confidence=1.5,
                action="NONE",
                priority="LOW",
            )

        # Invalid: < 0
        with pytest.raises(ValidationError):
            PredictionResponse(
                resource_id="test",
                cluster_id=0,
                cluster_name="NORMAL",
                confidence=-0.1,
                action="NONE",
                priority="LOW",
            )

    def test_cluster_id_range(self):
        """Test cluster_id must be between 0 and 4."""
        from scry.api.schemas import PredictionResponse

        # Valid: 0-4
        for i in range(5):
            PredictionResponse(
                resource_id="test",
                cluster_id=i,
                cluster_name="NORMAL",
                confidence=0.9,
                action="NONE",
                priority="LOW",
            )

        # Invalid: negative
        with pytest.raises(ValidationError):
            PredictionResponse(
                resource_id="test",
                cluster_id=-1,
                cluster_name="NORMAL",
                confidence=0.9,
                action="NONE",
                priority="LOW",
            )

        # Invalid: > 4
        with pytest.raises(ValidationError):
            PredictionResponse(
                resource_id="test",
                cluster_id=5,
                cluster_name="NORMAL",
                confidence=0.9,
                action="NONE",
                priority="LOW",
            )

    def test_serialization(self):
        """Test response serializes to dict correctly."""
        from scry.api.schemas import PredictionResponse

        response = PredictionResponse(
            resource_id="test-pod",
            cluster_id=2,
            cluster_name="PRE_FAILURE",
            confidence=0.87,
            action="DIAGNOSTIC",
            priority="HIGH",
        )

        data = response.model_dump()
        assert data["cluster_id"] == 2
        assert data["cluster_name"] == "PRE_FAILURE"
        assert data["confidence"] == 0.87


class TestHealthResponse:
    """Tests for HealthResponse schema."""

    def test_valid_health_response(self):
        """Test creating a valid health response."""
        from scry.api.schemas import HealthResponse

        response = HealthResponse(
            status="healthy",
            model_loaded=True,
            version="0.1.0",
        )

        assert response.status == "healthy"
        assert response.model_loaded is True
        assert response.version == "0.1.0"

    def test_unhealthy_status(self):
        """Test unhealthy status."""
        from scry.api.schemas import HealthResponse

        response = HealthResponse(
            status="unhealthy",
            model_loaded=False,
            version="0.1.0",
        )

        assert response.status == "unhealthy"
        assert response.model_loaded is False

    def test_serialization(self):
        """Test health response serializes correctly."""
        from scry.api.schemas import HealthResponse

        response = HealthResponse(
            status="healthy",
            model_loaded=True,
            version="0.1.0",
        )

        data = response.model_dump()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


class TestClusterInfo:
    """Tests for ClusterInfo schema."""

    def test_cluster_info_creation(self):
        """Test creating cluster info."""
        from scry.api.schemas import ClusterInfo

        info = ClusterInfo(
            id=0,
            name="NORMAL",
            action="NONE",
            priority="LOW",
            description="System operating normally",
        )

        assert info.id == 0
        assert info.name == "NORMAL"
        assert info.action == "NONE"
        assert info.priority == "LOW"
        assert info.description == "System operating normally"

    def test_get_all_clusters(self):
        """Test getting all cluster definitions."""
        from scry.api.schemas import get_cluster_info

        clusters = get_cluster_info()

        assert len(clusters) == 5
        assert clusters[0].name == "NORMAL"
        assert clusters[1].name == "PRE_SCALE"
        assert clusters[2].name == "PRE_FAILURE"
        assert clusters[3].name == "ACTIVE_DEGRADATION"
        assert clusters[4].name == "ANOMALY"

    def test_cluster_info_by_id(self):
        """Test getting cluster info by ID."""
        from scry.api.schemas import get_cluster_info

        clusters = get_cluster_info()

        # Check each cluster has correct action
        assert clusters[0].action == "NONE"
        assert clusters[1].action == "SCALE"
        assert clusters[2].action == "DIAGNOSTIC"
        assert clusters[3].action == "REMEDIATE"
        assert clusters[4].action == "ALERT"
