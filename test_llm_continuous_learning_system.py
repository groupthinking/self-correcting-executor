"""
Comprehensive unit tests for LLM Continuous Learning System.
Testing framework: pytest

This test suite covers:
- Initialization and configuration validation
- Data loading and validation
- Model training and fine-tuning (async operations)
- Feedback collection and processing
- Performance evaluation and metrics
- Error handling and edge cases
- Thread safety and concurrency
- Memory management
- Checkpoint operations
- Integration scenarios
"""

import pytest
import asyncio
import json
import threading
import time
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call
from datetime import datetime, timedelta
from typing import List, Dict, Any


# Import the module under test
try:
    from git.llm.continuous_learning_system import LLMContinuousLearningSystem
except ImportError:
    # Fallback import path
    from llm.continuous_learning_system import LLMContinuousLearningSystem


class TestLLMContinuousLearningSystemInitialization:
    """Test suite for system initialization and configuration."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock LLM model."""
        mock = Mock()
        mock.fine_tune = AsyncMock(return_value={"status": "success", "loss": 0.1})
        mock.evaluate = Mock(return_value={"accuracy": 0.85, "precision": 0.82, "recall": 0.88, "f1_score": 0.85})
        mock.save_checkpoint = Mock()
        mock.load_checkpoint = Mock()
        return mock

    @pytest.fixture
    def mock_data_loader(self):
        """Create a mock data loader."""
        mock = Mock()
        mock.load_training_data = Mock(return_value=[
            {"input": "What is Python?", "output": "Python is a programming language."},
            {"input": "Explain ML", "output": "Machine learning is a subset of AI."},
            {"input": "Define API", "output": "Application Programming Interface."}
        ])
        return mock

    @pytest.fixture
    def mock_feedback_collector(self):
        """Create a mock feedback collector."""
        mock = Mock()
        mock.collect_feedback = Mock(return_value=[
            {"query": "test query 1", "response": "test response 1", "rating": 5, "timestamp": datetime.now()},
            {"query": "test query 2", "response": "test response 2", "rating": 4, "timestamp": datetime.now()},
            {"query": "test query 3", "response": "test response 3", "rating": 3, "timestamp": datetime.now()}
        ])
        return mock

    @pytest.fixture
    def learning_system(self, mock_model, mock_data_loader, mock_feedback_collector):
        """Create a learning system instance for testing."""
        return LLMContinuousLearningSystem(
            model=mock_model,
            data_loader=mock_data_loader,
            feedback_collector=mock_feedback_collector
        )

    def test_successful_initialization_with_defaults(self, mock_model, mock_data_loader, mock_feedback_collector):
        """Test successful initialization with default parameters."""
        system = LLMContinuousLearningSystem(
            model=mock_model,
            data_loader=mock_data_loader,
            feedback_collector=mock_feedback_collector
        )
        
        assert system.model == mock_model
        assert system.data_loader == mock_data_loader
        assert system.feedback_collector == mock_feedback_collector
        assert system.learning_rate == 0.001
        assert system.batch_size == 16
        assert system.max_epochs == 10
        assert system.total_training_samples == 0
        assert system.total_feedback_samples == 0
        assert system.model_version == 1
        assert system.last_training_time is None
        assert not system._is_training

    def test_successful_initialization_with_custom_parameters(self, mock_model, mock_data_loader, mock_feedback_collector):
        """Test initialization with custom parameters."""
        system = LLMContinuousLearningSystem(
            model=mock_model,
            data_loader=mock_data_loader,
            feedback_collector=mock_feedback_collector,
            learning_rate=0.01,
            batch_size=32,
            max_epochs=20
        )
        
        assert system.learning_rate == 0.01
        assert system.batch_size == 32
        assert system.max_epochs == 20

    def test_initialization_fails_with_none_model(self, mock_data_loader, mock_feedback_collector):
        """Test that initialization fails when model is None."""
        with pytest.raises(ValueError, match="Model cannot be None"):
            LLMContinuousLearningSystem(
                model=None,
                data_loader=mock_data_loader,
                feedback_collector=mock_feedback_collector
            )

    def test_initialization_fails_with_invalid_learning_rate(self, mock_model, mock_data_loader, mock_feedback_collector):
        """Test that initialization fails with invalid learning rate."""
        with pytest.raises(ValueError, match="Learning rate must be positive"):
            LLMContinuousLearningSystem(
                model=mock_model,
                data_loader=mock_data_loader,
                feedback_collector=mock_feedback_collector,
                learning_rate=-0.01
            )

    def test_initialization_fails_with_zero_learning_rate(self, mock_model, mock_data_loader, mock_feedback_collector):
        """Test that initialization fails with zero learning rate."""
        with pytest.raises(ValueError, match="Learning rate must be positive"):
            LLMContinuousLearningSystem(
                model=mock_model,
                data_loader=mock_data_loader,
                feedback_collector=mock_feedback_collector,
                learning_rate=0.0
            )

    def test_initialization_fails_with_invalid_batch_size(self, mock_model, mock_data_loader, mock_feedback_collector):
        """Test that initialization fails with invalid batch size."""
        with pytest.raises(ValueError, match="Batch size must be positive"):
            LLMContinuousLearningSystem(
                model=mock_model,
                data_loader=mock_data_loader,
                feedback_collector=mock_feedback_collector,
                batch_size=0
            )

    @pytest.mark.parametrize("learning_rate,batch_size,max_epochs", [
        (0.001, 8, 5),
        (0.01, 16, 10),
        (0.1, 32, 15),
        (0.0001, 64, 20)
    ])
    def test_initialization_with_various_valid_parameters(self, mock_model, mock_data_loader, mock_feedback_collector,
                                                        learning_rate, batch_size, max_epochs):
        """Test initialization with various valid parameter combinations."""
        system = LLMContinuousLearningSystem(
            model=mock_model,
            data_loader=mock_data_loader,
            feedback_collector=mock_feedback_collector,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epochs=max_epochs
        )
        
        assert system.learning_rate == learning_rate
        assert system.batch_size == batch_size
        assert system.max_epochs == max_epochs


class TestLLMContinuousLearningSystemDataHandling:
    """Test suite for data loading and validation."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock LLM model."""
        return Mock()

    @pytest.fixture
    def mock_data_loader(self):
        """Create a mock data loader."""
        mock = Mock()
        mock.load_training_data = Mock(return_value=[
            {"input": "Sample input 1", "output": "Sample output 1"},
            {"input": "Sample input 2", "output": "Sample output 2"}
        ])
        return mock

    @pytest.fixture
    def mock_feedback_collector(self):
        """Create a mock feedback collector."""
        return Mock()

    @pytest.fixture
    def learning_system(self, mock_model, mock_data_loader, mock_feedback_collector):
        """Create a learning system instance for testing."""
        return LLMContinuousLearningSystem(
            model=mock_model,
            data_loader=mock_data_loader,
            feedback_collector=mock_feedback_collector
        )

    def test_load_training_data_success(self, learning_system):
        """Test successful loading of training data."""
        expected_data = [
            {"input": "Sample input 1", "output": "Sample output 1"},
            {"input": "Sample input 2", "output": "Sample output 2"}
        ]
        learning_system.data_loader.load_training_data.return_value = expected_data
        
        data = learning_system.load_training_data()
        
        assert data == expected_data
        learning_system.data_loader.load_training_data.assert_called_once()

    def test_load_training_data_empty_dataset(self, learning_system):
        """Test handling of empty training dataset."""
        learning_system.data_loader.load_training_data.return_value = []
        
        with pytest.raises(ValueError, match="Training data cannot be empty"):
            learning_system.load_training_data()

    def test_validate_training_data_valid_data(self, learning_system):
        """Test validation of valid training data."""
        valid_data = [
            {"input": "Valid input 1", "output": "Valid output 1"},
            {"input": "Valid input 2", "output": "Valid output 2"}
        ]
        
        result = learning_system.validate_training_data(valid_data)
        assert result is True

    def test_validate_training_data_missing_input_key(self, learning_system):
        """Test validation fails when input key is missing."""
        invalid_data = [{"output": "Valid output"}]
        
        with pytest.raises(ValueError, match="Invalid training data format"):
            learning_system.validate_training_data(invalid_data)

    def test_validate_training_data_missing_output_key(self, learning_system):
        """Test validation fails when output key is missing."""
        invalid_data = [{"input": "Valid input"}]
        
        with pytest.raises(ValueError, match="Invalid training data format"):
            learning_system.validate_training_data(invalid_data)

    def test_validate_training_data_empty_input(self, learning_system):
        """Test validation fails with empty input."""
        invalid_data = [{"input": "", "output": "Valid output"}]
        
        with pytest.raises(ValueError, match="Empty inputs or outputs not allowed"):
            learning_system.validate_training_data(invalid_data)

    def test_validate_training_data_empty_output(self, learning_system):
        """Test validation fails with empty output."""
        invalid_data = [{"input": "Valid input", "output": ""}]
        
        with pytest.raises(ValueError, match="Empty inputs or outputs not allowed"):
            learning_system.validate_training_data(invalid_data)

    def test_validate_training_data_none_input(self, learning_system):
        """Test validation fails with None input."""
        invalid_data = [{"input": None, "output": "Valid output"}]
        
        with pytest.raises(ValueError, match="Empty inputs or outputs not allowed"):
            learning_system.validate_training_data(invalid_data)

    def test_validate_training_data_input_too_long(self, learning_system):
        """Test validation fails when input exceeds maximum length."""
        long_input = "a" * (learning_system.max_input_length + 1)
        invalid_data = [{"input": long_input, "output": "Valid output"}]
        
        with pytest.raises(ValueError, match="Input exceeds maximum length"):
            learning_system.validate_training_data(invalid_data)

    def test_validate_training_data_non_dict_item(self, learning_system):
        """Test validation fails with non-dictionary items."""
        invalid_data = ["not a dictionary"]
        
        with pytest.raises(ValueError, match="Invalid training data format"):
            learning_system.validate_training_data(invalid_data)

    def test_validate_training_data_unicode_characters(self, learning_system):
        """Test validation handles unicode characters correctly."""
        unicode_data = [
            {"input": "Hello 世界! 🌍", "output": "Unicode test"},
            {"input": "Émojis: 😀😃😄", "output": "Emoji test"},
            {"input": "Special chars: @#$%^&*()", "output": "Symbols test"}
        ]
        
        result = learning_system.validate_training_data(unicode_data)
        assert result is True

    def test_create_training_batches_even_division(self, learning_system):
        """Test creating training batches with even division."""
        data = [{"input": f"input {i}", "output": f"output {i}"} for i in range(16)]
        learning_system.data_loader.load_training_data.return_value = data
        learning_system.batch_size = 8
        
        batches = learning_system.create_training_batches()
        
        assert len(batches) == 2
        assert len(batches[0]) == 8
        assert len(batches[1]) == 8

    def test_create_training_batches_uneven_division(self, learning_system):
        """Test creating training batches with uneven division."""
        data = [{"input": f"input {i}", "output": f"output {i}"} for i in range(10)]
        learning_system.data_loader.load_training_data.return_value = data
        learning_system.batch_size = 3
        
        batches = learning_system.create_training_batches()
        
        assert len(batches) == 4
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3
        assert len(batches[2]) == 3
        assert len(batches[3]) == 1

    def test_create_training_batches_single_batch(self, learning_system):
        """Test creating training batches when data fits in single batch."""
        data = [{"input": f"input {i}", "output": f"output {i}"} for i in range(5)]
        learning_system.data_loader.load_training_data.return_value = data
        learning_system.batch_size = 10
        
        batches = learning_system.create_training_batches()
        
        assert len(batches) == 1
        assert len(batches[0]) == 5


class TestLLMContinuousLearningSystemTraining:
    """Test suite for model training and fine-tuning operations."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock LLM model."""
        mock = Mock()
        mock.fine_tune = AsyncMock(return_value={"status": "success", "loss": 0.1, "accuracy": 0.9})
        mock.evaluate = Mock(return_value={"accuracy": 0.85, "precision": 0.82, "recall": 0.88, "f1_score": 0.85})
        return mock

    @pytest.fixture
    def mock_data_loader(self):
        """Create a mock data loader."""
        mock = Mock()
        mock.load_training_data = Mock(return_value=[
            {"input": "Training input 1", "output": "Training output 1"},
            {"input": "Training input 2", "output": "Training output 2"}
        ])
        return mock

    @pytest.fixture
    def mock_feedback_collector(self):
        """Create a mock feedback collector."""
        return Mock()

    @pytest.fixture
    def learning_system(self, mock_model, mock_data_loader, mock_feedback_collector):
        """Create a learning system instance for testing."""
        return LLMContinuousLearningSystem(
            model=mock_model,
            data_loader=mock_data_loader,
            feedback_collector=mock_feedback_collector
        )

    @pytest.mark.asyncio
    async def test_fine_tune_model_success(self, learning_system):
        """Test successful model fine-tuning."""
        initial_version = learning_system.model_version
        initial_samples = learning_system.total_training_samples
        
        result = await learning_system.fine_tune_model()
        
        assert result["status"] == "success"
        assert result["loss"] == 0.1
        assert result["accuracy"] == 0.9
        assert learning_system.model_version == initial_version + 1
        assert learning_system.total_training_samples == initial_samples + 2
        assert learning_system.last_training_time is not None
        assert not learning_system._is_training
        learning_system.model.fine_tune.assert_called_once()

    @pytest.mark.asyncio
    async def test_fine_tune_model_failure(self, learning_system):
        """Test handling of fine-tuning failures."""
        learning_system.model.fine_tune.side_effect = Exception("Fine-tuning failed")
        
        with pytest.raises(Exception, match="Fine-tuning failed"):
            await learning_system.fine_tune_model()
        
        assert not learning_system._is_training

    @pytest.mark.asyncio
    async def test_fine_tune_model_concurrent_training_prevention(self, learning_system):
        """Test prevention of concurrent training operations."""
        learning_system._is_training = True
        
        with pytest.raises(RuntimeError, match="Training already in progress"):
            await learning_system.fine_tune_model()

    @pytest.mark.asyncio
    async def test_fine_tune_model_updates_statistics(self, learning_system):
        """Test that fine-tuning updates system statistics correctly."""
        initial_time = learning_system.last_training_time
        initial_version = learning_system.model_version
        
        await learning_system.fine_tune_model()
        
        assert learning_system.last_training_time != initial_time
        assert learning_system.model_version == initial_version + 1
        assert learning_system.total_training_samples > 0

    def test_evaluate_model_performance_success(self, learning_system):
        """Test successful model performance evaluation."""
        expected_metrics = {"accuracy": 0.85, "precision": 0.82, "recall": 0.88, "f1_score": 0.85}
        learning_system.model.evaluate.return_value = expected_metrics
        
        metrics = learning_system.evaluate_model_performance()
        
        assert metrics == expected_metrics
        learning_system.model.evaluate.assert_called_once()

    def test_evaluate_model_performance_failure(self, learning_system):
        """Test handling of evaluation failures."""
        learning_system.model.evaluate.side_effect = Exception("Evaluation failed")
        initial_error_count = learning_system.error_count
        
        with pytest.raises(Exception, match="Evaluation failed"):
            learning_system.evaluate_model_performance()
        
        assert learning_system.error_count == initial_error_count + 1

    def test_calculate_learning_metrics_improvement(self, learning_system):
        """Test calculation of learning metrics with improvement."""
        old_metrics = {"accuracy": 0.80, "loss": 0.25}
        new_metrics = {"accuracy": 0.85, "loss": 0.20}
        
        improvement = learning_system.calculate_learning_metrics(old_metrics, new_metrics)
        
        assert improvement["accuracy_improvement"] == 0.05
        assert improvement["loss_reduction"] == 0.05

    def test_calculate_learning_metrics_degradation(self, learning_system):
        """Test calculation of learning metrics with performance degradation."""
        old_metrics = {"accuracy": 0.85, "loss": 0.20}
        new_metrics = {"accuracy": 0.80, "loss": 0.25}
        
        improvement = learning_system.calculate_learning_metrics(old_metrics, new_metrics)
        
        assert improvement["accuracy_improvement"] == -0.05
        assert improvement["loss_reduction"] == -0.05

    def test_calculate_learning_metrics_missing_keys(self, learning_system):
        """Test calculation with missing metric keys."""
        old_metrics = {}
        new_metrics = {"accuracy": 0.85}
        
        improvement = learning_system.calculate_learning_metrics(old_metrics, new_metrics)
        
        assert improvement["accuracy_improvement"] == 0.85
        assert improvement["loss_reduction"] == 0.0

    def test_simulate_long_training_success(self, learning_system):
        """Test simulation of long training session."""
        result = learning_system.simulate_long_training()
        
        assert result["status"] == "completed"


class TestLLMContinuousLearningSystemFeedback:
    """Test suite for feedback collection and processing."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock LLM model."""
        return Mock()

    @pytest.fixture
    def mock_data_loader(self):
        """Create a mock data loader."""
        return Mock()

    @pytest.fixture
    def mock_feedback_collector(self):
        """Create a mock feedback collector."""
        mock = Mock()
        mock.collect_feedback = Mock(return_value=[
            {"query": "test query 1", "response": "test response 1", "rating": 5, "timestamp": datetime.now()},
            {"query": "test query 2", "response": "test response 2", "rating": 4, "timestamp": datetime.now()},
            {"query": "test query 3", "response": "test response 3", "rating": 3, "timestamp": datetime.now()},
            {"query": "test query 4", "response": "test response 4", "rating": 2, "timestamp": datetime.now()},
            {"query": "test query 5", "response": "test response 5", "rating": 1, "timestamp": datetime.now()}
        ])
        return mock

    @pytest.fixture
    def learning_system(self, mock_model, mock_data_loader, mock_feedback_collector):
        """Create a learning system instance for testing."""
        return LLMContinuousLearningSystem(
            model=mock_model,
            data_loader=mock_data_loader,
            feedback_collector=mock_feedback_collector
        )

    @pytest.fixture
    def sample_feedback_data(self):
        """Sample feedback data for testing."""
        return [
            {"query": "What is AI?", "response": "AI is artificial intelligence.", "rating": 5, "timestamp": datetime.now()},
            {"query": "How does ML work?", "response": "ML uses algorithms to learn.", "rating": 4, "timestamp": datetime.now()},
            {"query": "What is deep learning?", "response": "Deep learning uses neural networks.", "rating": 3, "timestamp": datetime.now()},
            {"query": "Explain NLP", "response": "NLP processes natural language.", "rating": 2, "timestamp": datetime.now()},
            {"query": "What is computer vision?", "response": "Computer vision analyzes images.", "rating": 1, "timestamp": datetime.now()}
        ]

    def test_collect_feedback_success(self, learning_system):
        """Test successful feedback collection."""
        initial_feedback_count = learning_system.total_feedback_samples
        
        feedback = learning_system.collect_feedback()
        
        assert len(feedback) == 5
        assert learning_system.total_feedback_samples == initial_feedback_count + 5
        learning_system.feedback_collector.collect_feedback.assert_called_once()

    def test_collect_feedback_empty_results(self, learning_system):
        """Test handling of empty feedback collection."""
        learning_system.feedback_collector.collect_feedback.return_value = []
        
        feedback = learning_system.collect_feedback()
        
        assert feedback == []
        assert learning_system.total_feedback_samples == 0

    def test_filter_high_quality_feedback_default_threshold(self, learning_system, sample_feedback_data):
        """Test filtering high-quality feedback with default threshold (4)."""
        result = learning_system.filter_high_quality_feedback(sample_feedback_data)
        
        assert len(result) == 2  # Ratings 4 and 5
        assert all(item["rating"] >= 4 for item in result)

    def test_filter_high_quality_feedback_custom_threshold(self, learning_system, sample_feedback_data):
        """Test filtering high-quality feedback with custom threshold."""
        result = learning_system.filter_high_quality_feedback(sample_feedback_data, min_rating=3)
        
        assert len(result) == 3  # Ratings 3, 4, and 5
        assert all(item["rating"] >= 3 for item in result)

    def test_filter_high_quality_feedback_high_threshold(self, learning_system, sample_feedback_data):
        """Test filtering with high threshold that excludes all feedback."""
        result = learning_system.filter_high_quality_feedback(sample_feedback_data, min_rating=6)
        
        assert result == []

    def test_filter_high_quality_feedback_invalid_threshold(self, learning_system, sample_feedback_data):
        """Test filtering with invalid rating threshold."""
        with pytest.raises(ValueError, match="Invalid rating threshold"):
            learning_system.filter_high_quality_feedback(sample_feedback_data, min_rating=0)

    def test_filter_high_quality_feedback_negative_threshold(self, learning_system, sample_feedback_data):
        """Test filtering with negative rating threshold."""
        with pytest.raises(ValueError, match="Invalid rating threshold"):
            learning_system.filter_high_quality_feedback(sample_feedback_data, min_rating=-1)

    def test_filter_high_quality_feedback_missing_rating(self, learning_system):
        """Test filtering feedback items without rating."""
        feedback_without_rating = [
            {"query": "test", "response": "test response"},
            {"query": "test2", "response": "test response 2", "rating": 5}
        ]
        
        result = learning_system.filter_high_quality_feedback(feedback_without_rating, min_rating=4)
        
        assert len(result) == 1  # Only the one with rating 5
        assert result[0]["rating"] == 5

    @pytest.mark.parametrize("min_rating,expected_count", [
        (1, 5),  # All feedback
        (2, 4),  # Ratings 2, 3, 4, 5
        (3, 3),  # Ratings 3, 4, 5
        (4, 2),  # Ratings 4, 5
        (5, 1),  # Rating 5 only
    ])
    def test_filter_high_quality_feedback_various_thresholds(self, learning_system, sample_feedback_data, 
                                                           min_rating, expected_count):
        """Test filtering with various rating thresholds."""
        result = learning_system.filter_high_quality_feedback(sample_feedback_data, min_rating=min_rating)
        
        assert len(result) == expected_count
        assert all(item["rating"] >= min_rating for item in result)


class TestLLMContinuousLearningSystemContinuousLearning:
    """Test suite for continuous learning cycle operations."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock LLM model."""
        mock = Mock()
        mock.fine_tune = AsyncMock(return_value={"status": "success", "loss": 0.1})
        mock.evaluate = Mock(side_effect=[
            {"accuracy": 0.80, "loss": 0.25},  # Old metrics
            {"accuracy": 0.85, "loss": 0.20}   # New metrics
        ])
        return mock

    @pytest.fixture
    def mock_data_loader(self):
        """Create a mock data loader."""
        mock = Mock()
        mock.load_training_data = Mock(return_value=[
            {"input": "Training input", "output": "Training output"}
        ])
        return mock

    @pytest.fixture
    def mock_feedback_collector(self):
        """Create a mock feedback collector."""
        mock = Mock()
        mock.collect_feedback = Mock(return_value=[
            {"query": "high quality query 1", "response": "response 1", "rating": 5},
            {"query": "high quality query 2", "response": "response 2", "rating": 4},
            {"query": "low quality query", "response": "response 3", "rating": 2}
        ])
        return mock

    @pytest.fixture
    def learning_system(self, mock_model, mock_data_loader, mock_feedback_collector):
        """Create a learning system instance for testing."""
        return LLMContinuousLearningSystem(
            model=mock_model,
            data_loader=mock_data_loader,
            feedback_collector=mock_feedback_collector
        )

    @pytest.mark.asyncio
    async def test_continuous_learning_cycle_success(self, learning_system):
        """Test successful continuous learning cycle."""
        result = await learning_system.run_continuous_learning_cycle()
        
        assert result["status"] == "success"
        assert result["feedback_count"] == 3
        assert result["high_quality_count"] == 2
        assert "metrics" in result
        assert "improvement" in result
        assert result["improvement"]["accuracy_improvement"] == 0.05
        assert result["improvement"]["loss_reduction"] == 0.05

    @pytest.mark.asyncio
    async def test_continuous_learning_cycle_no_feedback(self, learning_system):
        """Test continuous learning cycle with no feedback available."""
        learning_system.feedback_collector.collect_feedback.return_value = []
        
        result = await learning_system.run_continuous_learning_cycle()
        
        assert result["status"] == "skipped"
        assert result["reason"] == "No feedback available"

    @pytest.mark.asyncio
    async def test_continuous_learning_cycle_no_high_quality_feedback(self, learning_system):
        """Test continuous learning cycle with no high-quality feedback."""
        learning_system.feedback_collector.collect_feedback.return_value = [
            {"query": "low quality query 1", "response": "response 1", "rating": 2},
            {"query": "low quality query 2", "response": "response 2", "rating": 1}
        ]
        
        result = await learning_system.run_continuous_learning_cycle()
        
        assert result["status"] == "skipped"
        assert result["reason"] == "No high-quality feedback"

    @pytest.mark.asyncio
    async def test_continuous_learning_cycle_training_failure(self, learning_system):
        """Test continuous learning cycle with training failure."""
        learning_system.model.fine_tune.side_effect = Exception("Training failed")
        
        with pytest.raises(Exception, match="Training failed"):
            await learning_system.run_continuous_learning_cycle()

    @pytest.mark.asyncio
    async def test_continuous_learning_cycle_evaluation_failure(self, learning_system):
        """Test continuous learning cycle with evaluation failure."""
        learning_system.model.evaluate.side_effect = Exception("Evaluation failed")
        
        with pytest.raises(Exception, match="Evaluation failed"):
            await learning_system.run_continuous_learning_cycle()


class TestLLMContinuousLearningSystemCheckpoints:
    """Test suite for checkpoint operations."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock LLM model."""
        mock = Mock()
        mock.save_checkpoint = Mock()
        mock.load_checkpoint = Mock()
        return mock

    @pytest.fixture
    def mock_data_loader(self):
        """Create a mock data loader."""
        return Mock()

    @pytest.fixture
    def mock_feedback_collector(self):
        """Create a mock feedback collector."""
        return Mock()

    @pytest.fixture
    def learning_system(self, mock_model, mock_data_loader, mock_feedback_collector):
        """Create a learning system instance for testing."""
        return LLMContinuousLearningSystem(
            model=mock_model,
            data_loader=mock_data_loader,
            feedback_collector=mock_feedback_collector
        )

    def test_save_model_checkpoint_success(self, learning_system):
        """Test successful model checkpoint saving."""
        checkpoint_path = "/tmp/test_checkpoint.pkl"
        
        learning_system.save_model_checkpoint(checkpoint_path)
        
        learning_system.model.save_checkpoint.assert_called_once_with(checkpoint_path)

    def test_load_model_checkpoint_success(self, learning_system):
        """Test successful model checkpoint loading."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            checkpoint_path = temp_file.name
            temp_file.write(b"dummy checkpoint data")
        
        try:
            learning_system.load_model_checkpoint(checkpoint_path)
            learning_system.model.load_checkpoint.assert_called_once_with(checkpoint_path)
        finally:
            os.unlink(checkpoint_path)

    def test_load_model_checkpoint_file_not_found(self, learning_system):
        """Test loading checkpoint when file doesn't exist."""
        nonexistent_path = "/tmp/nonexistent_checkpoint.pkl"
        
        with pytest.raises(FileNotFoundError, match=f"Checkpoint file not found: {nonexistent_path}"):
            learning_system.load_model_checkpoint(nonexistent_path)

    def test_save_checkpoint_with_various_paths(self, learning_system):
        """Test saving checkpoints with various path formats."""
        paths = [
            "/tmp/checkpoint1.pkl",
            "./checkpoint2.pkl",
            "checkpoint3.pkl",
            "/path/to/deep/directory/checkpoint4.pkl"
        ]
        
        for path in paths:
            learning_system.save_model_checkpoint(path)
            learning_system.model.save_checkpoint.assert_called_with(path)


class TestLLMContinuousLearningSystemStatistics:
    """Test suite for system statistics and monitoring."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock LLM model."""
        return Mock()

    @pytest.fixture
    def mock_data_loader(self):
        """Create a mock data loader."""
        return Mock()

    @pytest.fixture
    def mock_feedback_collector(self):
        """Create a mock feedback collector."""
        return Mock()

    @pytest.fixture
    def learning_system(self, mock_model, mock_data_loader, mock_feedback_collector):
        """Create a learning system instance for testing."""
        return LLMContinuousLearningSystem(
            model=mock_model,
            data_loader=mock_data_loader,
            feedback_collector=mock_feedback_collector
        )

    def test_get_system_statistics_initial_state(self, learning_system):
        """Test getting system statistics in initial state."""
        stats = learning_system.get_system_statistics()
        
        assert stats["total_training_samples"] == 0
        assert stats["total_feedback_samples"] == 0
        assert stats["model_version"] == 1
        assert stats["last_training_time"] is None
        assert stats["error_count"] == 0
        assert stats["is_training"] is False

    def test_get_system_statistics_after_updates(self, learning_system):
        """Test getting system statistics after updates."""
        # Simulate some activity
        learning_system.total_training_samples = 100
        learning_system.total_feedback_samples = 50
        learning_system.model_version = 3
        learning_system.error_count = 2
        learning_system.last_training_time = datetime.now()
        learning_system._is_training = True
        
        stats = learning_system.get_system_statistics()
        
        assert stats["total_training_samples"] == 100
        assert stats["total_feedback_samples"] == 50
        assert stats["model_version"] == 3
        assert stats["error_count"] == 2
        assert stats["last_training_time"] is not None
        assert stats["is_training"] is True

    def test_reset_learning_history(self, learning_system):
        """Test resetting learning history."""
        # Set some values first
        learning_system.total_training_samples = 100
        learning_system.total_feedback_samples = 50
        learning_system.error_count = 5
        learning_system.last_training_time = datetime.now()
        
        learning_system.reset_learning_history()
        
        stats = learning_system.get_system_statistics()
        assert stats["total_training_samples"] == 0
        assert stats["total_feedback_samples"] == 0
        assert stats["error_count"] == 0
        assert stats["last_training_time"] is None

    def test_memory_management(self, learning_system):
        """Test memory management functions."""
        # These are basic tests since the actual implementation is simulated
        initial_memory = learning_system.get_memory_usage()
        assert isinstance(initial_memory, int)
        assert initial_memory > 0
        
        learning_system.cleanup_memory()
        # Should not raise any exceptions


class TestLLMContinuousLearningSystemConfiguration:
    """Test suite for configuration validation."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock LLM model."""
        return Mock()

    @pytest.fixture
    def mock_data_loader(self):
        """Create a mock data loader."""
        return Mock()

    @pytest.fixture
    def mock_feedback_collector(self):
        """Create a mock feedback collector."""
        return Mock()

    @pytest.fixture
    def learning_system(self, mock_model, mock_data_loader, mock_feedback_collector):
        """Create a learning system instance for testing."""
        return LLMContinuousLearningSystem(
            model=mock_model,
            data_loader=mock_data_loader,
            feedback_collector=mock_feedback_collector
        )

    def test_validate_configuration_valid_config(self, learning_system):
        """Test validation of valid configuration."""
        valid_config = {
            "learning_rate": 0.01,
            "batch_size": 16,
            "max_epochs": 10
        }
        
        result = learning_system.validate_configuration(valid_config)
        assert result is True

    def test_validate_configuration_missing_learning_rate(self, learning_system):
        """Test validation fails when learning_rate is missing."""
        invalid_config = {
            "batch_size": 16,
            "max_epochs": 10
        }
        
        result = learning_system.validate_configuration(invalid_config)
        assert result is False

    def test_validate_configuration_missing_batch_size(self, learning_system):
        """Test validation fails when batch_size is missing."""
        invalid_config = {
            "learning_rate": 0.01,
            "max_epochs": 10
        }
        
        result = learning_system.validate_configuration(invalid_config)
        assert result is False

    def test_validate_configuration_missing_max_epochs(self, learning_system):
        """Test validation fails when max_epochs is missing."""
        invalid_config = {
            "learning_rate": 0.01,
            "batch_size": 16
        }
        
        result = learning_system.validate_configuration(invalid_config)
        assert result is False

    def test_validate_configuration_negative_learning_rate(self, learning_system):
        """Test validation fails with negative learning rate."""
        invalid_config = {
            "learning_rate": -0.01,
            "batch_size": 16,
            "max_epochs": 10
        }
        
        result = learning_system.validate_configuration(invalid_config)
        assert result is False

    def test_validate_configuration_zero_batch_size(self, learning_system):
        """Test validation fails with zero batch size."""
        invalid_config = {
            "learning_rate": 0.01,
            "batch_size": 0,
            "max_epochs": 10
        }
        
        result = learning_system.validate_configuration(invalid_config)
        assert result is False

    def test_validate_configuration_negative_max_epochs(self, learning_system):
        """Test validation fails with negative max epochs."""
        invalid_config = {
            "learning_rate": 0.01,
            "batch_size": 16,
            "max_epochs": -5
        }
        
        result = learning_system.validate_configuration(invalid_config)
        assert result is False

    @pytest.mark.parametrize("config,expected", [
        ({"learning_rate": 0.001, "batch_size": 8, "max_epochs": 5}, True),
        ({"learning_rate": 0.1, "batch_size": 32, "max_epochs": 20}, True),
        ({"learning_rate": 0, "batch_size": 16, "max_epochs": 10}, False),
        ({"learning_rate": 0.01, "batch_size": -1, "max_epochs": 10}, False),
        ({"learning_rate": 0.01, "batch_size": 16, "max_epochs": 0}, False),
    ])
    def test_validate_configuration_various_values(self, learning_system, config, expected):
        """Test configuration validation with various value combinations."""
        result = learning_system.validate_configuration(config)
        assert result == expected


class TestLLMContinuousLearningSystemConcurrency:
    """Test suite for concurrency and thread safety."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock LLM model."""
        mock = Mock()
        mock.fine_tune = AsyncMock(return_value={"status": "success"})
        mock.evaluate = Mock(return_value={"accuracy": 0.85})
        return mock

    @pytest.fixture
    def mock_data_loader(self):
        """Create a mock data loader."""
        mock = Mock()
        mock.load_training_data = Mock(return_value=[
            {"input": "test", "output": "test"}
        ])
        return mock

    @pytest.fixture
    def mock_feedback_collector(self):
        """Create a mock feedback collector."""
        return Mock()

    @pytest.fixture
    def learning_system(self, mock_model, mock_data_loader, mock_feedback_collector):
        """Create a learning system instance for testing."""
        return LLMContinuousLearningSystem(
            model=mock_model,
            data_loader=mock_data_loader,
            feedback_collector=mock_feedback_collector
        )

    def test_thread_safety_statistics_access(self, learning_system):
        """Test thread-safe access to system statistics."""
        results = []
        errors = []
        
        def worker():
            try:
                for _ in range(10):
                    stats = learning_system.get_system_statistics()
                    results.append(stats)
                    time.sleep(0.001)  # Small delay to increase chance of race conditions
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=worker) for _ in range(5)]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0  # No exceptions should occur
        assert len(results) == 50  # 5 threads * 10 calls each
        
        # All results should be valid dictionaries
        for result in results:
            assert isinstance(result, dict)
            assert "total_training_samples" in result

    @pytest.mark.asyncio
    async def test_training_lock_mechanism(self, learning_system):
        """Test that training lock prevents concurrent training."""
        # Start first training (will succeed)
        training_task1 = asyncio.create_task(learning_system.fine_tune_model())
        
        # Wait a bit to ensure first training starts
        await asyncio.sleep(0.01)
        
        # Try to start second training (should fail)
        with pytest.raises(RuntimeError, match="Training already in progress"):
            await learning_system.fine_tune_model()
        
        # Wait for first training to complete
        await training_task1

    def test_concurrent_statistics_updates(self, learning_system):
        """Test concurrent updates to statistics."""
        def update_worker():
            for i in range(100):
                learning_system.total_training_samples += 1
                learning_system.total_feedback_samples += 1
        
        threads = [threading.Thread(target=update_worker) for _ in range(3)]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # Note: This test may have race conditions in a real scenario
        # but it helps identify potential issues
        assert learning_system.total_training_samples <= 300
        assert learning_system.total_feedback_samples <= 300


class TestLLMContinuousLearningSystemEdgeCases:
    """Test suite for edge cases and error scenarios."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock LLM model."""
        return Mock()

    @pytest.fixture
    def mock_data_loader(self):
        """Create a mock data loader."""
        return Mock()

    @pytest.fixture
    def mock_feedback_collector(self):
        """Create a mock feedback collector."""
        return Mock()

    @pytest.fixture
    def learning_system(self, mock_model, mock_data_loader, mock_feedback_collector):
        """Create a learning system instance for testing."""
        return LLMContinuousLearningSystem(
            model=mock_model,
            data_loader=mock_data_loader,
            feedback_collector=mock_feedback_collector
        )

    def test_edge_case_very_large_input(self, learning_system):
        """Test handling of very large input data."""
        large_input = "x" * 50000  # Very large input
        large_data = [{"input": large_input, "output": "output"}]
        
        learning_system.max_input_length = 1000  # Set limit
        
        with pytest.raises(ValueError, match="Input exceeds maximum length"):
            learning_system.validate_training_data(large_data)

    def test_edge_case_empty_strings(self, learning_system):
        """Test handling of empty strings in data."""
        empty_data = [
            {"input": "", "output": "valid"},
            {"input": "valid", "output": ""},
            {"input": "   ", "output": "valid"}  # Whitespace only
        ]
        
        with pytest.raises(ValueError, match="Empty inputs or outputs not allowed"):
            learning_system.validate_training_data(empty_data)

    def test_edge_case_none_values(self, learning_system):
        """Test handling of None values in data."""
        none_data = [
            {"input": None, "output": "valid"},
            {"input": "valid", "output": None}
        ]
        
        with pytest.raises(ValueError, match="Empty inputs or outputs not allowed"):
            learning_system.validate_training_data(none_data)

    def test_edge_case_extreme_ratings(self, learning_system):
        """Test handling of extreme rating values."""
        extreme_feedback = [
            {"query": "test", "response": "test", "rating": 1000},  # Very high
            {"query": "test", "response": "test", "rating": -100},  # Negative
            {"query": "test", "response": "test", "rating": 0}      # Zero
        ]
        
        # Should handle extreme values gracefully
        result = learning_system.filter_high_quality_feedback(extreme_feedback, min_rating=5)
        assert len(result) == 1  # Only the rating of 1000 should pass

    def test_edge_case_unicode_and_emoji_handling(self, learning_system):
        """Test proper handling of unicode characters and emojis."""
        unicode_data = [
            {"input": "Hello 世界! 🌍", "output": "Unicode response 🚀"},
            {"input": "Émojis: 😀😃😄😁", "output": "Emoji response"},
            {"input": "Math symbols: ∑∏∫∆", "output": "Math response"},
            {"input": "Special: ñáéíóú", "output": "Accented response"}
        ]
        
        # Should handle unicode gracefully
        result = learning_system.validate_training_data(unicode_data)
        assert result is True

    def test_edge_case_very_small_batch_size(self, learning_system):
        """Test handling of very small batch sizes."""
        data = [{"input": f"input {i}", "output": f"output {i}"} for i in range(10)]
        learning_system.data_loader.load_training_data.return_value = data
        learning_system.batch_size = 1
        
        batches = learning_system.create_training_batches()
        
        assert len(batches) == 10
        assert all(len(batch) == 1 for batch in batches)

    def test_edge_case_batch_size_larger_than_data(self, learning_system):
        """Test handling when batch size is larger than available data."""
        data = [{"input": "single input", "output": "single output"}]
        learning_system.data_loader.load_training_data.return_value = data
        learning_system.batch_size = 100
        
        batches = learning_system.create_training_batches()
        
        assert len(batches) == 1
        assert len(batches[0]) == 1

    def test_error_count_incrementation(self, learning_system):
        """Test that error count is properly incremented."""
        learning_system.model.evaluate.side_effect = Exception("Test error")
        initial_count = learning_system.error_count
        
        try:
            learning_system.evaluate_model_performance()
        except Exception:
            pass
        
        assert learning_system.error_count == initial_count + 1


# Integration and Performance Test Markers
class TestLLMContinuousLearningSystemIntegration:
    """Integration tests for the system (marked appropriately)."""

    @pytest.mark.integration
    def test_end_to_end_learning_pipeline(self):
        """Test complete end-to-end learning pipeline."""
        pytest.skip("Integration test - requires real model and data components")

    @pytest.mark.integration
    def test_real_model_fine_tuning(self):
        """Test with actual model fine-tuning."""
        pytest.skip("Integration test - requires real LLM model")

    @pytest.mark.integration
    def test_database_persistence(self):
        """Test persistence of training data and feedback."""
        pytest.skip("Integration test - requires database setup")


class TestLLMContinuousLearningSystemPerformance:
    """Performance tests for the system (marked appropriately)."""

    @pytest.mark.performance
    def test_large_dataset_processing(self):
        """Test processing of large datasets."""
        pytest.skip("Performance test - requires large dataset and extended runtime")

    @pytest.mark.performance
    def test_memory_usage_under_load(self):
        """Test memory usage under high load."""
        pytest.skip("Performance test - requires memory profiling tools")

    @pytest.mark.performance
    def test_concurrent_training_performance(self):
        """Test performance under concurrent operations."""
        pytest.skip("Performance test - requires load testing setup")


# Utility functions for creating test data
def create_sample_training_data(size: int) -> List[Dict[str, str]]:
    """Create sample training data for testing."""
    return [
        {"input": f"Sample input {i}", "output": f"Sample output {i}"}
        for i in range(size)
    ]


def create_sample_feedback_data(size: int, rating_range: Tuple[int, int] = (1, 5)) -> List[Dict[str, Any]]:
    """Create sample feedback data for testing."""
    min_rating, max_rating = rating_range
    return [
        {
            "query": f"Query {i}",
            "response": f"Response {i}",
            "rating": min_rating + (i % (max_rating - min_rating + 1)),
            "timestamp": datetime.now() - timedelta(days=i)
        }
        for i in range(size)
    ]


# Pytest configuration
pytestmark = [
    pytest.mark.unit,  # Mark all tests as unit tests by default
]

# Test configuration for different environments
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "slow: Slow-running tests")

class TestLLMContinuousLearningSystemAdvancedErrorHandling:
    """Advanced error handling and exception scenarios."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock LLM model with various failure modes."""
        mock = Mock()
        mock.fine_tune = AsyncMock()
        mock.evaluate = Mock()
        mock.save_checkpoint = Mock()
        mock.load_checkpoint = Mock()
        return mock

    @pytest.fixture
    def mock_data_loader(self):
        """Create a mock data loader with failure scenarios."""
        mock = Mock()
        mock.load_training_data = Mock()
        return mock

    @pytest.fixture
    def mock_feedback_collector(self):
        """Create a mock feedback collector with failure scenarios."""
        mock = Mock()
        mock.collect_feedback = Mock()
        return mock

    @pytest.fixture
    def learning_system(self, mock_model, mock_data_loader, mock_feedback_collector):
        """Create a learning system instance for testing."""
        return LLMContinuousLearningSystem(
            model=mock_model,
            data_loader=mock_data_loader,
            feedback_collector=mock_feedback_collector
        )

    def test_data_loader_raises_ioerror(self, learning_system):
        """Test handling when data loader raises IOError."""
        learning_system.data_loader.load_training_data.side_effect = IOError("Cannot read data file")
        
        with pytest.raises(IOError, match="Cannot read data file"):
            learning_system.load_training_data()

    def test_data_loader_raises_permission_error(self, learning_system):
        """Test handling when data loader raises PermissionError."""
        learning_system.data_loader.load_training_data.side_effect = PermissionError("Access denied")
        
        with pytest.raises(PermissionError, match="Access denied"):
            learning_system.load_training_data()

    def test_data_loader_raises_memory_error(self, learning_system):
        """Test handling when data loader raises MemoryError."""
        learning_system.data_loader.load_training_data.side_effect = MemoryError("Out of memory")
        
        with pytest.raises(MemoryError, match="Out of memory"):
            learning_system.load_training_data()

    @pytest.mark.asyncio
    async def test_model_fine_tune_timeout(self, learning_system):
        """Test handling of model fine-tuning timeout."""
        learning_system.model.fine_tune.side_effect = asyncio.TimeoutError("Training timed out")
        
        with pytest.raises(asyncio.TimeoutError, match="Training timed out"):
            await learning_system.fine_tune_model()

    @pytest.mark.asyncio
    async def test_model_fine_tune_cancelled(self, learning_system):
        """Test handling of cancelled fine-tuning operation."""
        learning_system.model.fine_tune.side_effect = asyncio.CancelledError("Training cancelled")
        
        with pytest.raises(asyncio.CancelledError, match="Training cancelled"):
            await learning_system.fine_tune_model()

    def test_feedback_collector_network_error(self, learning_system):
        """Test handling of network errors during feedback collection."""
        learning_system.feedback_collector.collect_feedback.side_effect = ConnectionError("Network unreachable")
        
        with pytest.raises(ConnectionError, match="Network unreachable"):
            learning_system.collect_feedback()

    def test_feedback_collector_json_decode_error(self, learning_system):
        """Test handling of JSON decode errors during feedback collection."""
        learning_system.feedback_collector.collect_feedback.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        
        with pytest.raises(json.JSONDecodeError):
            learning_system.collect_feedback()

    def test_model_evaluation_cuda_error(self, learning_system):
        """Test handling of CUDA errors during model evaluation."""
        learning_system.model.evaluate.side_effect = RuntimeError("CUDA out of memory")
        
        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            learning_system.evaluate_model_performance()

    def test_checkpoint_save_disk_full_error(self, learning_system):
        """Test handling of disk full error during checkpoint save."""
        learning_system.model.save_checkpoint.side_effect = OSError("No space left on device")
        
        with pytest.raises(OSError, match="No space left on device"):
            learning_system.save_model_checkpoint("/tmp/checkpoint.pkl")

    def test_checkpoint_load_corrupted_file(self, learning_system):
        """Test handling of corrupted checkpoint file."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"corrupted data")
            checkpoint_path = temp_file.name
        
        learning_system.model.load_checkpoint.side_effect = EOFError("Corrupted checkpoint file")
        
        try:
            with pytest.raises(EOFError, match="Corrupted checkpoint file"):
                learning_system.load_model_checkpoint(checkpoint_path)
        finally:
            os.unlink(checkpoint_path)

    def test_multiple_sequential_errors(self, learning_system):
        """Test handling of multiple sequential errors."""
        learning_system.model.evaluate.side_effect = [
            RuntimeError("First error"),
            ValueError("Second error"),
            Exception("Third error")
        ]
        
        initial_error_count = learning_system.error_count
        
        for i in range(3):
            with pytest.raises(Exception):
                learning_system.evaluate_model_performance()
        
        assert learning_system.error_count == initial_error_count + 3

    @pytest.mark.parametrize("exception_type,message", [
        (ValueError, "Invalid parameter"),
        (TypeError, "Type mismatch"),
        (AttributeError, "Missing attribute"),
        (KeyError, "Missing key"),
        (IndexError, "Index out of range"),
    ])
    def test_various_exception_types(self, learning_system, exception_type, message):
        """Test handling of various exception types."""
        learning_system.model.evaluate.side_effect = exception_type(message)
        
        with pytest.raises(exception_type, match=message):
            learning_system.evaluate_model_performance()


class TestLLMContinuousLearningSystemAdvancedValidation:
    """Advanced validation and data integrity tests."""

    @pytest.fixture
    def mock_model(self):
        return Mock()

    @pytest.fixture
    def mock_data_loader(self):
        return Mock()

    @pytest.fixture
    def mock_feedback_collector(self):
        return Mock()

    @pytest.fixture
    def learning_system(self, mock_model, mock_data_loader, mock_feedback_collector):
        return LLMContinuousLearningSystem(
            model=mock_model,
            data_loader=mock_data_loader,
            feedback_collector=mock_feedback_collector
        )

    def test_validate_data_with_nested_structures(self, learning_system):
        """Test validation of training data with nested structures."""
        nested_data = [
            {
                "input": {"text": "Hello", "metadata": {"lang": "en"}},
                "output": "Hi there!"
            }
        ]
        
        # Should handle nested structures appropriately
        with pytest.raises(ValueError, match="Invalid training data format"):
            learning_system.validate_training_data(nested_data)

    def test_validate_data_with_numeric_values(self, learning_system):
        """Test validation with numeric input/output values."""
        numeric_data = [
            {"input": 123, "output": "Number: 123"},
            {"input": "Calculate: 2+2", "output": 4}
        ]
        
        with pytest.raises(ValueError, match="Invalid training data format"):
            learning_system.validate_training_data(numeric_data)

    def test_validate_data_with_boolean_values(self, learning_system):
        """Test validation with boolean input/output values."""
        boolean_data = [
            {"input": True, "output": "Boolean value"},
            {"input": "Is this true?", "output": False}
        ]
        
        with pytest.raises(ValueError, match="Invalid training data format"):
            learning_system.validate_training_data(boolean_data)

    def test_validate_data_with_list_values(self, learning_system):
        """Test validation with list input/output values."""
        list_data = [
            {"input": ["item1", "item2"], "output": "List items"},
            {"input": "What are the items?", "output": ["a", "b", "c"]}
        ]
        
        with pytest.raises(ValueError, match="Invalid training data format"):
            learning_system.validate_training_data(list_data)

    def test_validate_data_with_extra_keys(self, learning_system):
        """Test validation with extra keys in data."""
        extra_keys_data = [
            {
                "input": "Valid input",
                "output": "Valid output",
                "extra_field": "Should be ignored",
                "metadata": {"version": 1}
            }
        ]
        
        # Should validate successfully, ignoring extra keys
        result = learning_system.validate_training_data(extra_keys_data)
        assert result is True

    def test_validate_feedback_with_invalid_timestamp(self, learning_system):
        """Test feedback validation with invalid timestamp."""
        invalid_feedback = [
            {
                "query": "test",
                "response": "test",
                "rating": 5,
                "timestamp": "invalid_timestamp"
            }
        ]
        
        # Should handle invalid timestamp gracefully
        result = learning_system.filter_high_quality_feedback(invalid_feedback)
        assert len(result) == 1  # Should still include the feedback

    def test_validate_feedback_with_missing_fields(self, learning_system):
        """Test feedback validation with missing fields."""
        incomplete_feedback = [
            {"query": "test", "rating": 5},  # Missing response
            {"response": "test", "rating": 4},  # Missing query
            {"query": "test", "response": "test"}  # Missing rating
        ]
        
        result = learning_system.filter_high_quality_feedback(incomplete_feedback)
        assert len(result) == 0  # Should filter out incomplete feedback

    @pytest.mark.parametrize("data_size", [1, 10, 100, 1000])
    def test_validate_data_various_sizes(self, learning_system, data_size):
        """Test validation with various data sizes."""
        data = [
            {"input": f"Input {i}", "output": f"Output {i}"}
            for i in range(data_size)
        ]
        
        result = learning_system.validate_training_data(data)
        assert result is True

    def test_validate_data_with_whitespace_variations(self, learning_system):
        """Test validation with various whitespace patterns."""
        whitespace_data = [
            {"input": "  Valid input  ", "output": "Valid output"},
            {"input": "Valid input", "output": "  Valid output  "},
            {"input": "\tTabbed input\t", "output": "Valid output"},
            {"input": "Valid input", "output": "\nNewline output\n"}
        ]
        
        result = learning_system.validate_training_data(whitespace_data)
        assert result is True

    def test_validate_data_with_sql_injection_patterns(self, learning_system):
        """Test validation with SQL injection-like patterns."""
        sql_injection_data = [
            {"input": "'; DROP TABLE users; --", "output": "SQL injection attempt"},
            {"input": "1' OR '1'='1", "output": "Another injection attempt"},
            {"input": "UNION SELECT * FROM passwords", "output": "Union attack"}
        ]
        
        result = learning_system.validate_training_data(sql_injection_data)
        assert result is True  # Should accept as valid text

    def test_validate_data_with_xss_patterns(self, learning_system):
        """Test validation with XSS-like patterns."""
        xss_data = [
            {"input": "<script>alert('xss')</script>", "output": "XSS attempt"},
            {"input": "javascript:alert(1)", "output": "JavaScript injection"},
            {"input": "<img src=x onerror=alert(1)>", "output": "Image XSS"}
        ]
        
        result = learning_system.validate_training_data(xss_data)
        assert result is True  # Should accept as valid text


class TestLLMContinuousLearningSystemAdvancedConcurrency:
    """Advanced concurrency and race condition tests."""

    @pytest.fixture
    def mock_model(self):
        mock = Mock()
        mock.fine_tune = AsyncMock(return_value={"status": "success"})
        mock.evaluate = Mock(return_value={"accuracy": 0.85})
        return mock

    @pytest.fixture
    def mock_data_loader(self):
        mock = Mock()
        mock.load_training_data = Mock(return_value=[
            {"input": "test", "output": "test"}
        ])
        return mock

    @pytest.fixture
    def mock_feedback_collector(self):
        return Mock()

    @pytest.fixture
    def learning_system(self, mock_model, mock_data_loader, mock_feedback_collector):
        return LLMContinuousLearningSystem(
            model=mock_model,
            data_loader=mock_data_loader,
            feedback_collector=mock_feedback_collector
        )

    def test_concurrent_statistics_read_write(self, learning_system):
        """Test concurrent reading and writing of statistics."""
        results = []
        errors = []
        
        def reader():
            try:
                for _ in range(20):
                    stats = learning_system.get_system_statistics()
                    results.append(stats['total_training_samples'])
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        def writer():
            try:
                for i in range(20):
                    learning_system.total_training_samples = i
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        reader_threads = [threading.Thread(target=reader) for _ in range(3)]
        writer_threads = [threading.Thread(target=writer) for _ in range(2)]
        
        all_threads = reader_threads + writer_threads
        
        for t in all_threads:
            t.start()
        
        for t in all_threads:
            t.join()
        
        assert len(errors) == 0
        assert len(results) == 60  # 3 readers * 20 calls each

    @pytest.mark.asyncio
    async def test_multiple_async_operations(self, learning_system):
        """Test multiple async operations running concurrently."""
        # Create multiple async tasks
        tasks = []
        
        for i in range(5):
            # Each task will try to fine-tune but only one should succeed
            task = asyncio.create_task(learning_system.fine_tune_model())
            tasks.append(task)
            await asyncio.sleep(0.001)  # Small delay between task creation
        
        # Wait for all tasks to complete (some will fail with RuntimeError)
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful and failed operations
        successful = sum(1 for r in results if isinstance(r, dict) and r.get('status') == 'success')
        failed = sum(1 for r in results if isinstance(r, RuntimeError))
        
        # Should have exactly one success and multiple failures
        assert successful == 1
        assert failed == 4

    def test_memory_statistics_under_load(self, learning_system):
        """Test memory statistics under concurrent load."""
        def memory_worker():
            for _ in range(50):
                memory_usage = learning_system.get_memory_usage()
                assert memory_usage > 0
                learning_system.cleanup_memory()
                time.sleep(0.001)
        
        threads = [threading.Thread(target=memory_worker) for _ in range(5)]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # Should not raise any exceptions

    def test_checkpoint_operations_under_load(self, learning_system):
        """Test checkpoint operations under concurrent load."""
        checkpoint_paths = [f"/tmp/checkpoint_{i}.pkl" for i in range(10)]
        errors = []
        
        def checkpoint_worker(path):
            try:
                learning_system.save_model_checkpoint(path)
                time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=checkpoint_worker, args=(path,)) for path in checkpoint_paths]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0  # No errors should occur
        assert learning_system.model.save_checkpoint.call_count == 10

    @pytest.mark.asyncio
    async def test_async_training_with_interruption(self, learning_system):
        """Test async training with various interruption scenarios."""
        # Set up a slow training process
        async def slow_training():
            await asyncio.sleep(0.1)
            return {"status": "success"}
        
        learning_system.model.fine_tune = AsyncMock(side_effect=slow_training)
        
        # Start training
        training_task = asyncio.create_task(learning_system.fine_tune_model())
        
        # Wait a bit then try to interrupt
        await asyncio.sleep(0.05)
        
        # Try to start another training (should fail)
        with pytest.raises(RuntimeError, match="Training already in progress"):
            await learning_system.fine_tune_model()
        
        # Wait for original training to complete
        result = await training_task
        assert result["status"] == "success"


class TestLLMContinuousLearningSystemAdvancedBatching:
    """Advanced batching and data processing tests."""

    @pytest.fixture
    def mock_model(self):
        return Mock()

    @pytest.fixture
    def mock_data_loader(self):
        return Mock()

    @pytest.fixture
    def mock_feedback_collector(self):
        return Mock()

    @pytest.fixture
    def learning_system(self, mock_model, mock_data_loader, mock_feedback_collector):
        return LLMContinuousLearningSystem(
            model=mock_model,
            data_loader=mock_data_loader,
            feedback_collector=mock_feedback_collector
        )

    @pytest.mark.parametrize("data_size,batch_size,expected_batches", [
        (0, 16, 0),
        (1, 16, 1),
        (15, 16, 1),
        (16, 16, 1),
        (17, 16, 2),
        (32, 16, 2),
        (33, 16, 3),
        (100, 7, 15),  # 100/7 = 14.28... = 15 batches
        (1000, 1, 1000),
    ])
    def test_batch_creation_various_combinations(self, learning_system, data_size, batch_size, expected_batches):
        """Test batch creation with various data size and batch size combinations."""
        data = [{"input": f"input {i}", "output": f"output {i}"} for i in range(data_size)]
        learning_system.data_loader.load_training_data.return_value = data
        learning_system.batch_size = batch_size
        
        if data_size == 0:
            # Should handle empty data appropriately
            with pytest.raises(ValueError, match="Training data cannot be empty"):
                learning_system.create_training_batches()
        else:
            batches = learning_system.create_training_batches()
            assert len(batches) == expected_batches
            
            # Verify total items in all batches equals original data size
            total_items = sum(len(batch) for batch in batches)
            assert total_items == data_size

    def test_batch_content_integrity(self, learning_system):
        """Test that batch content maintains data integrity."""
        original_data = [
            {"input": f"input {i}", "output": f"output {i}", "id": i}
            for i in range(25)
        ]
        learning_system.data_loader.load_training_data.return_value = original_data
        learning_system.batch_size = 7
        
        batches = learning_system.create_training_batches()
        
        # Reconstruct data from batches
        reconstructed_data = []
        for batch in batches:
            reconstructed_data.extend(batch)
        
        # Verify all original data is preserved
        assert len(reconstructed_data) == len(original_data)
        
        # Verify each item is preserved exactly
        for i, original_item in enumerate(original_data):
            assert original_item in reconstructed_data

    def test_batch_processing_with_duplicates(self, learning_system):
        """Test batch processing with duplicate data."""
        duplicate_data = [
            {"input": "duplicate input", "output": "duplicate output"}
        ] * 10
        
        learning_system.data_loader.load_training_data.return_value = duplicate_data
        learning_system.batch_size = 3
        
        batches = learning_system.create_training_batches()
        
        # Should handle duplicates without issues
        assert len(batches) == 4  # 10/3 = 3.33... = 4 batches
        
        # Verify all duplicates are preserved
        total_items = sum(len(batch) for batch in batches)
        assert total_items == 10

    def test_batch_processing_with_varying_sizes(self, learning_system):
        """Test batch processing with data items of varying sizes."""
        varying_data = [
            {"input": "short", "output": "short"},
            {"input": "medium length input text", "output": "medium length output text"},
            {"input": "very long input text that contains many words and characters", 
             "output": "very long output text that also contains many words and characters"},
            {"input": "a" * 1000, "output": "b" * 1000}  # Very long strings
        ]
        
        learning_system.data_loader.load_training_data.return_value = varying_data
        learning_system.batch_size = 2
        
        batches = learning_system.create_training_batches()
        
        assert len(batches) == 2
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2

    def test_batch_memory_efficiency(self, learning_system):
        """Test batch creation memory efficiency."""
        # Create large dataset
        large_data = [
            {"input": f"input {i} " * 100, "output": f"output {i} " * 100}
            for i in range(1000)
        ]
        
        learning_system.data_loader.load_training_data.return_value = large_data
        learning_system.batch_size = 50
        
        # Should create batches without memory issues
        batches = learning_system.create_training_batches()
        
        assert len(batches) == 20  # 1000/50 = 20
        assert all(len(batch) == 50 for batch in batches)


class TestLLMContinuousLearningSystemAdvancedMetrics:
    """Advanced metrics calculation and analysis tests."""

    @pytest.fixture
    def mock_model(self):
        return Mock()

    @pytest.fixture
    def mock_data_loader(self):
        return Mock()

    @pytest.fixture
    def mock_feedback_collector(self):
        return Mock()

    @pytest.fixture
    def learning_system(self, mock_model, mock_data_loader, mock_feedback_collector):
        return LLMContinuousLearningSystem(
            model=mock_model,
            data_loader=mock_data_loader,
            feedback_collector=mock_feedback_collector
        )

    @pytest.mark.parametrize("old_metrics,new_metrics,expected_accuracy_improvement,expected_loss_reduction", [
        ({"accuracy": 0.8, "loss": 0.2}, {"accuracy": 0.9, "loss": 0.1}, 0.1, 0.1),
        ({"accuracy": 0.5, "loss": 0.5}, {"accuracy": 0.6, "loss": 0.4}, 0.1, 0.1),
        ({"accuracy": 0.9, "loss": 0.1}, {"accuracy": 0.8, "loss": 0.2}, -0.1, -0.1),
        ({"accuracy": 0.0, "loss": 1.0}, {"accuracy": 1.0, "loss": 0.0}, 1.0, 1.0),
        ({"accuracy": 0.5, "loss": 0.5}, {"accuracy": 0.5, "loss": 0.5}, 0.0, 0.0),
    ])
    def test_metrics_calculation_various_scenarios(self, learning_system, old_metrics, new_metrics, 
                                                 expected_accuracy_improvement, expected_loss_reduction):
        """Test metrics calculation with various improvement/degradation scenarios."""
        improvement = learning_system.calculate_learning_metrics(old_metrics, new_metrics)
        
        assert abs(improvement["accuracy_improvement"] - expected_accuracy_improvement) < 1e-6
        assert abs(improvement["loss_reduction"] - expected_loss_reduction) < 1e-6

    def test_metrics_with_additional_metrics(self, learning_system):
        """Test metrics calculation with additional metric types."""
        old_metrics = {
            "accuracy": 0.8,
            "loss": 0.2,
            "precision": 0.75,
            "recall": 0.85,
            "f1_score": 0.80
        }
        new_metrics = {
            "accuracy": 0.85,
            "loss": 0.15,
            "precision": 0.80,
            "recall": 0.90,
            "f1_score": 0.85
        }
        
        improvement = learning_system.calculate_learning_metrics(old_metrics, new_metrics)
        
        assert improvement["accuracy_improvement"] == 0.05
        assert improvement["loss_reduction"] == 0.05
        # Should handle additional metrics gracefully

    def test_metrics_with_missing_values(self, learning_system):
        """Test metrics calculation with missing values."""
        old_metrics = {"accuracy": 0.8}
        new_metrics = {"loss": 0.15}
        
        improvement = learning_system.calculate_learning_metrics(old_metrics, new_metrics)
        
        # Should handle missing values gracefully
        assert improvement["accuracy_improvement"] == 0.0
        assert improvement["loss_reduction"] == 0.0

    def test_metrics_with_nan_values(self, learning_system):
        """Test metrics calculation with NaN values."""
        old_metrics = {"accuracy": float('nan'), "loss": 0.2}
        new_metrics = {"accuracy": 0.85, "loss": float('nan')}
        
        improvement = learning_system.calculate_learning_metrics(old_metrics, new_metrics)
        
        # Should handle NaN values gracefully
        assert improvement["accuracy_improvement"] == 0.85
        assert improvement["loss_reduction"] == 0.0

    def test_metrics_with_infinity_values(self, learning_system):
        """Test metrics calculation with infinity values."""
        old_metrics = {"accuracy": 0.8, "loss": float('inf')}
        new_metrics = {"accuracy": float('inf'), "loss": 0.15}
        
        improvement = learning_system.calculate_learning_metrics(old_metrics, new_metrics)
        
        # Should handle infinity values gracefully
        assert improvement["accuracy_improvement"] == float('inf')
        assert improvement["loss_reduction"] == float('inf')

    def test_metrics_with_negative_values(self, learning_system):
        """Test metrics calculation with negative values."""
        old_metrics = {"accuracy": -0.5, "loss": -0.3}
        new_metrics = {"accuracy": 0.8, "loss": 0.2}
        
        improvement = learning_system.calculate_learning_metrics(old_metrics, new_metrics)
        
        assert improvement["accuracy_improvement"] == 1.3
        assert improvement["loss_reduction"] == -0.5  # Loss increased

    @pytest.mark.parametrize("metric_type", ["accuracy", "loss", "precision", "recall", "f1_score"])
    def test_individual_metric_improvements(self, learning_system, metric_type):
        """Test calculation of individual metric improvements."""
        old_metrics = {metric_type: 0.7}
        new_metrics = {metric_type: 0.8}
        
        improvement = learning_system.calculate_learning_metrics(old_metrics, new_metrics)
        
        if metric_type == "accuracy":
            assert improvement["accuracy_improvement"] == 0.1
        elif metric_type == "loss":
            assert improvement["loss_reduction"] == 0.1
        # Other metrics should not affect the standard calculations
        assert "accuracy_improvement" in improvement
        assert "loss_reduction" in improvement


class TestLLMContinuousLearningSystemAdvancedMemoryManagement:
    """Advanced memory management and resource handling tests."""

    @pytest.fixture
    def mock_model(self):
        return Mock()

    @pytest.fixture
    def mock_data_loader(self):
        return Mock()

    @pytest.fixture
    def mock_feedback_collector(self):
        return Mock()

    @pytest.fixture
    def learning_system(self, mock_model, mock_data_loader, mock_feedback_collector):
        return LLMContinuousLearningSystem(
            model=mock_model,
            data_loader=mock_data_loader,
            feedback_collector=mock_feedback_collector
        )

    def test_memory_usage_tracking(self, learning_system):
        """Test memory usage tracking functionality."""
        initial_memory = learning_system.get_memory_usage()
        
        # Simulate memory usage by creating data
        learning_system.total_training_samples = 10000
        learning_system.total_feedback_samples = 5000
        
        # Memory usage should remain consistent (since it's mocked)
        current_memory = learning_system.get_memory_usage()
        assert isinstance(current_memory, int)
        assert current_memory > 0

    def test_memory_cleanup_operations(self, learning_system):
        """Test memory cleanup operations."""
        # Set up some data
        learning_system.total_training_samples = 1000
        learning_system.total_feedback_samples = 500
        
        # Cleanup should not raise exceptions
        learning_system.cleanup_memory()
        
        # System should still be functional after cleanup
        stats = learning_system.get_system_statistics()
        assert isinstance(stats, dict)

    def test_memory_operations_under_stress(self, learning_system):
        """Test memory operations under stress conditions."""
        # Simulate high memory usage scenario
        for i in range(100):
            learning_system.get_memory_usage()
            learning_system.cleanup_memory()
            
            # Update counters to simulate activity
            learning_system.total_training_samples += 10
            learning_system.total_feedback_samples += 5
        
        # Should handle stress without issues
        final_stats = learning_system.get_system_statistics()
        assert final_stats["total_training_samples"] == 1000
        assert final_stats["total_feedback_samples"] == 500

    def test_memory_with_large_datasets(self, learning_system):
        """Test memory handling with large datasets."""
        # Simulate large dataset processing
        large_data = [
            {"input": f"Large input {i} " * 1000, "output": f"Large output {i} " * 1000}
            for i in range(10)  # Smaller number to avoid actual memory issues
        ]
        
        learning_system.data_loader.load_training_data.return_value = large_data
        learning_system.batch_size = 2
        
        # Should handle large data without memory errors
        batches = learning_system.create_training_batches()
        assert len(batches) == 5
        
        # Memory operations should work
        memory_usage = learning_system.get_memory_usage()
        assert memory_usage > 0
        
        learning_system.cleanup_memory()


class TestLLMContinuousLearningSystemAdvancedConfiguration:
    """Advanced configuration and parameter validation tests."""

    @pytest.fixture
    def mock_model(self):
        return Mock()

    @pytest.fixture
    def mock_data_loader(self):
        return Mock()

    @pytest.fixture
    def mock_feedback_collector(self):
        return Mock()

    @pytest.fixture
    def learning_system(self, mock_model, mock_data_loader, mock_feedback_collector):
        return LLMContinuousLearningSystem(
            model=mock_model,
            data_loader=mock_data_loader,
            feedback_collector=mock_feedback_collector
        )

    @pytest.mark.parametrize("config", [
        {},  # Empty config
        {"learning_rate": 0.01},  # Partial config
        {"batch_size": 32},  # Partial config
        {"max_epochs": 15},  # Partial config
        {"learning_rate": 0.01, "batch_size": 32},  # Two parameters
        {"extra_param": "value"},  # Extra parameter
    ])
    def test_configuration_validation_edge_cases(self, learning_system, config):
        """Test configuration validation with various edge cases."""
        if len(config) == 0 or any(key not in ["learning_rate", "batch_size", "max_epochs"] for key in config):
            # Should fail validation for empty or incomplete configs
            result = learning_system.validate_configuration(config)
            assert result is False
        else:
            # Should pass for valid partial configs
            result = learning_system.validate_configuration(config)
            # Result depends on whether all required keys are present

    def test_configuration_with_extreme_values(self, learning_system):
        """Test configuration with extreme but valid values."""
        extreme_configs = [
            {"learning_rate": 1e-10, "batch_size": 1, "max_epochs": 1},
            {"learning_rate": 0.9, "batch_size": 1024, "max_epochs": 1000},
            {"learning_rate": 0.5, "batch_size": 2048, "max_epochs": 10000},
        ]
        
        for config in extreme_configs:
            result = learning_system.validate_configuration(config)
            assert result is True

    def test_configuration_with_string_values(self, learning_system):
        """Test configuration with string values (should fail)."""
        string_config = {
            "learning_rate": "0.01",
            "batch_size": "16",
            "max_epochs": "10"
        }
        
        result = learning_system.validate_configuration(string_config)
        assert result is False

    def test_configuration_with_float_batch_size(self, learning_system):
        """Test configuration with float batch size (should fail)."""
        float_config = {
            "learning_rate": 0.01,
            "batch_size": 16.5,
            "max_epochs": 10
        }
        
        result = learning_system.validate_configuration(float_config)
        assert result is False

    def test_configuration_with_nested_dict(self, learning_system):
        """Test configuration with nested dictionary values."""
        nested_config = {
            "learning_rate": 0.01,
            "batch_size": 16,
            "max_epochs": 10,
            "advanced": {"optimizer": "adam", "scheduler": "cosine"}
        }
        
        result = learning_system.validate_configuration(nested_config)
        # Should handle nested structures gracefully
        assert result is True

    def test_configuration_validation_consistency(self, learning_system):
        """Test that configuration validation is consistent across calls."""
        valid_config = {
            "learning_rate": 0.01,
            "batch_size": 16,
            "max_epochs": 10
        }
        
        # Multiple calls should return the same result
        results = [learning_system.validate_configuration(valid_config) for _ in range(10)]
        assert all(results)
        assert len(set(results)) == 1  # All results should be the same

    @pytest.mark.parametrize("num_calls", [1, 10, 100])
    def test_configuration_validation_performance(self, learning_system, num_calls):
        """Test configuration validation performance with multiple calls."""
        config = {
            "learning_rate": 0.01,
            "batch_size": 16,
            "max_epochs": 10
        }
        
        start_time = time.time()
        for _ in range(num_calls):
            learning_system.validate_configuration(config)
        end_time = time.time()
        
        # Should complete quickly regardless of number of calls
        assert end_time - start_time < 1.0  # Should complete within 1 second


# Additional utility test functions for comprehensive coverage
class TestLLMContinuousLearningSystemUtilities:
    """Test utility functions and helper methods."""

    @pytest.fixture
    def mock_model(self):
        return Mock()

    @pytest.fixture
    def mock_data_loader(self):
        return Mock()

    @pytest.fixture
    def mock_feedback_collector(self):
        return Mock()

    @pytest.fixture
    def learning_system(self, mock_model, mock_data_loader, mock_feedback_collector):
        return LLMContinuousLearningSystem(
            model=mock_model,
            data_loader=mock_data_loader,
            feedback_collector=mock_feedback_collector
        )

    def test_system_state_consistency(self, learning_system):
        """Test that system state remains consistent across operations."""
        initial_state = {
            "training_samples": learning_system.total_training_samples,
            "feedback_samples": learning_system.total_feedback_samples,
            "model_version": learning_system.model_version,
            "error_count": learning_system.error_count
        }
        
        # Perform various operations
        stats = learning_system.get_system_statistics()
        memory = learning_system.get_memory_usage()
        learning_system.cleanup_memory()
        
        # State should remain consistent
        final_state = {
            "training_samples": learning_system.total_training_samples,
            "feedback_samples": learning_system.total_feedback_samples,
            "model_version": learning_system.model_version,
            "error_count": learning_system.error_count
        }
        
        assert initial_state == final_state

    def test_system_initialization_idempotency(self, mock_model, mock_data_loader, mock_feedback_collector):
        """Test that multiple system initializations are idempotent."""
        system1 = LLMContinuousLearningSystem(
            model=mock_model,
            data_loader=mock_data_loader,
            feedback_collector=mock_feedback_collector
        )
        
        system2 = LLMContinuousLearningSystem(
            model=mock_model,
            data_loader=mock_data_loader,
            feedback_collector=mock_feedback_collector
        )
        
        # Both systems should have identical initial states
        stats1 = system1.get_system_statistics()
        stats2 = system2.get_system_statistics()
        
        # Remove instance-specific fields for comparison
        comparable_stats1 = {k: v for k, v in stats1.items() if k != 'last_training_time'}
        comparable_stats2 = {k: v for k, v in stats2.items() if k != 'last_training_time'}
        
        assert comparable_stats1 == comparable_stats2

    def test_error_handling_doesnt_affect_system_state(self, learning_system):
        """Test that error handling doesn't corrupt system state."""
        initial_stats = learning_system.get_system_statistics()
        initial_error_count = initial_stats["error_count"]
        
        # Cause an error
        learning_system.model.evaluate.side_effect = Exception("Test error")
        
        try:
            learning_system.evaluate_model_performance()
        except Exception:
            pass
        
        # Check that only error count increased
        final_stats = learning_system.get_system_statistics()
        assert final_stats["error_count"] == initial_error_count + 1
        
        # Other stats should remain unchanged
        for key in ["total_training_samples", "total_feedback_samples", "model_version"]:
            assert final_stats[key] == initial_stats[key]

    def test_system_statistics_completeness(self, learning_system):
        """Test that system statistics contain all expected fields."""
        stats = learning_system.get_system_statistics()
        
        expected_fields = [
            "total_training_samples",
            "total_feedback_samples",
            "model_version",
            "last_training_time",
            "error_count",
            "is_training"
        ]
        
        for field in expected_fields:
            assert field in stats, f"Missing field: {field}"

    def test_system_statistics_types(self, learning_system):
        """Test that system statistics have correct types."""
        stats = learning_system.get_system_statistics()
        
        assert isinstance(stats["total_training_samples"], int)
        assert isinstance(stats["total_feedback_samples"], int)
        assert isinstance(stats["model_version"], int)
        assert isinstance(stats["error_count"], int)
        assert isinstance(stats["is_training"], bool)
        # last_training_time can be None or datetime


# Pytest configuration additions
pytest.mark.usefixtures("mock_model", "mock_data_loader", "mock_feedback_collector")

# Additional markers for the new test classes
pytestmark.extend([
    pytest.mark.advanced,  # Mark advanced tests
    pytest.mark.comprehensive,  # Mark comprehensive tests
])
