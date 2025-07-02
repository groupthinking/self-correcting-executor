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

class TestLLMContinuousLearningSystemAdvancedScenarios:
    """Advanced test scenarios for comprehensive coverage."""

    @pytest.fixture
    def mock_model_with_complex_behavior(self):
        """Create a mock model with complex behavior patterns."""
        mock = Mock()
        mock.fine_tune = AsyncMock()
        mock.evaluate = Mock()
        mock.save_checkpoint = Mock()
        mock.load_checkpoint = Mock()
        mock.get_model_size = Mock(return_value=1000000)  # 1MB model
        mock.get_training_progress = Mock(return_value={"epoch": 5, "loss": 0.15})
        return mock

    @pytest.fixture
    def mock_data_loader_with_streaming(self):
        """Create a mock data loader that supports streaming."""
        mock = Mock()
        mock.load_training_data = Mock()
        mock.stream_training_data = Mock()
        mock.get_data_statistics = Mock(return_value={"total_samples": 1000, "avg_length": 150})
        return mock

    @pytest.fixture
    def mock_feedback_collector_with_analytics(self):
        """Create a mock feedback collector with analytics capabilities."""
        mock = Mock()
        mock.collect_feedback = Mock()
        mock.get_feedback_analytics = Mock(return_value={"avg_rating": 4.2, "total_feedback": 500})
        mock.filter_feedback_by_date = Mock()
        return mock

    @pytest.fixture
    def advanced_learning_system(self, mock_model_with_complex_behavior, 
                                mock_data_loader_with_streaming, 
                                mock_feedback_collector_with_analytics):
        """Create an advanced learning system instance for testing."""
        return LLMContinuousLearningSystem(
            model=mock_model_with_complex_behavior,
            data_loader=mock_data_loader_with_streaming,
            feedback_collector=mock_feedback_collector_with_analytics,
            learning_rate=0.001,
            batch_size=32,
            max_epochs=15
        )

    @pytest.mark.parametrize("learning_rate,batch_size,max_epochs,expected_valid", [
        (0.0001, 1, 1, True),     # Minimum valid values
        (1.0, 1024, 1000, True),  # Maximum reasonable values
        (0.5, 64, 50, True),      # Mid-range values
        (2.0, 128, 25, True),     # High learning rate
        (1e-6, 2, 5, True),       # Very small learning rate
        (0, 16, 10, False),       # Zero learning rate (invalid)
        (0.01, 0, 10, False),     # Zero batch size (invalid)
        (0.01, 16, 0, False),     # Zero epochs (invalid)
        (-0.01, 16, 10, False),   # Negative learning rate (invalid)
        (0.01, -16, 10, False),   # Negative batch size (invalid)
        (0.01, 16, -10, False),   # Negative epochs (invalid)
    ])
    def test_extensive_parameter_validation(self, mock_model_with_complex_behavior,
                                          mock_data_loader_with_streaming,
                                          mock_feedback_collector_with_analytics,
                                          learning_rate, batch_size, max_epochs, expected_valid):
        """Test extensive parameter validation with various combinations."""
        if expected_valid:
            system = LLMContinuousLearningSystem(
                model=mock_model_with_complex_behavior,
                data_loader=mock_data_loader_with_streaming,
                feedback_collector=mock_feedback_collector_with_analytics,
                learning_rate=learning_rate,
                batch_size=batch_size,
                max_epochs=max_epochs
            )
            assert system.learning_rate == learning_rate
            assert system.batch_size == batch_size
            assert system.max_epochs == max_epochs
        else:
            with pytest.raises(ValueError):
                LLMContinuousLearningSystem(
                    model=mock_model_with_complex_behavior,
                    data_loader=mock_data_loader_with_streaming,
                    feedback_collector=mock_feedback_collector_with_analytics,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    max_epochs=max_epochs
                )

    @pytest.mark.asyncio
    async def test_fine_tuning_with_progress_tracking(self, advanced_learning_system):
        """Test fine-tuning with detailed progress tracking."""
        # Setup progressive fine-tuning results
        progress_sequence = [
            {"status": "in_progress", "epoch": 1, "loss": 0.5, "accuracy": 0.7},
            {"status": "in_progress", "epoch": 2, "loss": 0.4, "accuracy": 0.75},
            {"status": "in_progress", "epoch": 3, "loss": 0.3, "accuracy": 0.8},
            {"status": "success", "epoch": 3, "loss": 0.25, "accuracy": 0.85}
        ]
        
        advanced_learning_system.model.fine_tune.return_value = progress_sequence[-1]
        advanced_learning_system.model.get_training_progress.side_effect = progress_sequence
        
        result = await advanced_learning_system.fine_tune_model()
        
        assert result["status"] == "success"
        assert result["loss"] == 0.25
        assert result["accuracy"] == 0.85
        advanced_learning_system.model.fine_tune.assert_called_once()

    @pytest.mark.asyncio
    async def test_fine_tuning_with_early_stopping(self, advanced_learning_system):
        """Test fine-tuning with early stopping mechanism."""
        # Simulate early stopping due to no improvement
        advanced_learning_system.model.fine_tune.return_value = {
            "status": "early_stopped", 
            "reason": "no_improvement", 
            "final_epoch": 7,
            "best_loss": 0.15,
            "best_accuracy": 0.88
        }
        
        result = await advanced_learning_system.fine_tune_model()
        
        assert result["status"] == "early_stopped"
        assert result["reason"] == "no_improvement"
        assert result["final_epoch"] == 7
        assert result["best_loss"] == 0.15

    @pytest.mark.asyncio
    async def test_fine_tuning_with_timeout(self, advanced_learning_system):
        """Test fine-tuning behavior with timeout scenarios."""
        # Simulate timeout during training
        advanced_learning_system.model.fine_tune.side_effect = asyncio.TimeoutError("Training timeout")
        
        with pytest.raises(asyncio.TimeoutError, match="Training timeout"):
            await advanced_learning_system.fine_tune_model()
        
        # Verify training state is properly reset
        assert not advanced_learning_system._is_training

    def test_complex_data_validation_scenarios(self, advanced_learning_system):
        """Test complex data validation scenarios."""
        # Test mixed valid and invalid data
        mixed_data = [
            {"input": "Valid input 1", "output": "Valid output 1"},
            {"input": "", "output": "Invalid empty input"},  # Invalid
            {"input": "Valid input 2", "output": "Valid output 2"},
            {"input": "Valid input 3", "output": ""},  # Invalid empty output
            {"input": "Valid input 4", "output": "Valid output 4"},
        ]
        
        with pytest.raises(ValueError, match="Empty inputs or outputs not allowed"):
            advanced_learning_system.validate_training_data(mixed_data)

    def test_data_validation_with_special_characters(self, advanced_learning_system):
        """Test data validation with various special character scenarios."""
        special_char_data = [
            {"input": "Input with\nnewlines\nand\ttabs", "output": "Output with special chars"},
            {"input": "Input with \r carriage returns", "output": "Normal output"},
            {"input": "Input with null\0characters", "output": "Output response"},
            {"input": "Input with \"quotes\" and 'apostrophes'", "output": "Quoted output"},
            {"input": "Input with <HTML> & XML entities", "output": "Encoded output"},
        ]
        
        # Should handle special characters appropriately
        result = advanced_learning_system.validate_training_data(special_char_data)
        assert result is True

    def test_batch_creation_with_mixed_data_sizes(self, advanced_learning_system):
        """Test batch creation with varying data sample sizes."""
        # Create data with varying input/output lengths
        varied_data = []
        for i in range(50):
            input_length = (i % 10) + 1  # Vary from 1 to 10 words
            output_length = ((i + 5) % 8) + 1  # Vary from 1 to 8 words
            varied_data.append({
                "input": " ".join([f"input_word_{j}" for j in range(input_length)]),
                "output": " ".join([f"output_word_{j}" for j in range(output_length)])
            })
        
        advanced_learning_system.data_loader.load_training_data.return_value = varied_data
        advanced_learning_system.batch_size = 7
        
        batches = advanced_learning_system.create_training_batches()
        
        # Check batch distribution
        total_samples = sum(len(batch) for batch in batches)
        assert total_samples == 50
        assert len(batches) == 8  # 50 / 7 = 7 full batches + 1 partial
        assert len(batches[-1]) == 1  # Last batch should have 1 sample

    @pytest.mark.parametrize("feedback_size,min_rating,expected_high_quality", [
        (100, 4, 40),   # Assuming 40% have rating >= 4
        (50, 3, 30),    # Assuming 60% have rating >= 3
        (25, 5, 5),     # Assuming 20% have rating = 5
        (10, 2, 8),     # Assuming 80% have rating >= 2
        (0, 4, 0),      # No feedback
    ])
    def test_feedback_filtering_with_various_distributions(self, advanced_learning_system,
                                                         feedback_size, min_rating, expected_high_quality):
        """Test feedback filtering with various rating distributions."""
        # Generate feedback with realistic rating distribution
        feedback_data = []
        for i in range(feedback_size):
            # Create a distribution where higher ratings are less common
            if i < feedback_size * 0.2:
                rating = 5
            elif i < feedback_size * 0.4:
                rating = 4
            elif i < feedback_size * 0.6:
                rating = 3
            elif i < feedback_size * 0.8:
                rating = 2
            else:
                rating = 1
            
            feedback_data.append({
                "query": f"Query {i}",
                "response": f"Response {i}",
                "rating": rating,
                "timestamp": datetime.now()
            })
        
        result = advanced_learning_system.filter_high_quality_feedback(feedback_data, min_rating=min_rating)
        
        # Allow for some variance in expected count due to distribution approximation
        assert abs(len(result) - expected_high_quality) <= 2
        assert all(item["rating"] >= min_rating for item in result)

    def test_system_statistics_with_comprehensive_metrics(self, advanced_learning_system):
        """Test system statistics with comprehensive metrics tracking."""
        # Set up comprehensive system state
        advanced_learning_system.total_training_samples = 1500
        advanced_learning_system.total_feedback_samples = 750
        advanced_learning_system.model_version = 5
        advanced_learning_system.error_count = 3
        advanced_learning_system.last_training_time = datetime.now() - timedelta(hours=2)
        advanced_learning_system._is_training = False
        
        stats = advanced_learning_system.get_system_statistics()
        
        # Verify all expected metrics are present
        expected_keys = [
            "total_training_samples", "total_feedback_samples", "model_version",
            "last_training_time", "error_count", "is_training"
        ]
        
        for key in expected_keys:
            assert key in stats
        
        assert stats["total_training_samples"] == 1500
        assert stats["total_feedback_samples"] == 750
        assert stats["model_version"] == 5
        assert stats["error_count"] == 3
        assert stats["is_training"] is False

    @pytest.mark.asyncio
    async def test_continuous_learning_with_incremental_improvement(self, advanced_learning_system):
        """Test continuous learning cycle with incremental improvements."""
        # Setup incremental improvement scenario
        metrics_sequence = [
            {"accuracy": 0.80, "precision": 0.78, "recall": 0.82, "f1_score": 0.80, "loss": 0.25},
            {"accuracy": 0.83, "precision": 0.81, "recall": 0.85, "f1_score": 0.83, "loss": 0.22},
        ]
        
        advanced_learning_system.model.evaluate.side_effect = metrics_sequence
        advanced_learning_system.model.fine_tune.return_value = {"status": "success", "loss": 0.22}
        advanced_learning_system.feedback_collector.collect_feedback.return_value = [
            {"query": "test1", "response": "resp1", "rating": 5, "timestamp": datetime.now()},
            {"query": "test2", "response": "resp2", "rating": 4, "timestamp": datetime.now()},
        ]
        
        result = await advanced_learning_system.run_continuous_learning_cycle()
        
        assert result["status"] == "success"
        assert result["improvement"]["accuracy_improvement"] == 0.03
        assert result["improvement"]["loss_reduction"] == 0.03
        assert "metrics" in result

    def test_checkpoint_operations_with_metadata(self, advanced_learning_system):
        """Test checkpoint operations with metadata handling."""
        checkpoint_path = "/tmp/test_checkpoint_with_metadata.pkl"
        
        # Test saving with metadata
        advanced_learning_system.save_model_checkpoint(checkpoint_path)
        advanced_learning_system.model.save_checkpoint.assert_called_once_with(checkpoint_path)
        
        # Verify mock was called correctly
        call_args = advanced_learning_system.model.save_checkpoint.call_args
        assert call_args[0][0] == checkpoint_path

    def test_memory_management_under_stress(self, advanced_learning_system):
        """Test memory management under stress conditions."""
        # Simulate memory usage tracking
        initial_memory = advanced_learning_system.get_memory_usage()
        
        # Simulate memory-intensive operations
        for _ in range(10):
            advanced_learning_system.cleanup_memory()
        
        # Memory management should not raise exceptions
        final_memory = advanced_learning_system.get_memory_usage()
        assert isinstance(final_memory, int)
        assert final_memory > 0

    @pytest.mark.parametrize("error_scenario", [
        "model_unavailable",
        "data_corrupted", 
        "network_timeout",
        "insufficient_memory",
        "permission_denied"
    ])
    def test_error_handling_for_various_failures(self, advanced_learning_system, error_scenario):
        """Test error handling for various failure scenarios."""
        error_messages = {
            "model_unavailable": "Model service unavailable",
            "data_corrupted": "Training data corruption detected",
            "network_timeout": "Network connection timeout",
            "insufficient_memory": "Insufficient memory for operation",
            "permission_denied": "Permission denied for file access"
        }
        
        error_message = error_messages[error_scenario]
        advanced_learning_system.model.evaluate.side_effect = Exception(error_message)
        
        initial_error_count = advanced_learning_system.error_count
        
        with pytest.raises(Exception, match=error_message):
            advanced_learning_system.evaluate_model_performance()
        
        assert advanced_learning_system.error_count == initial_error_count + 1

    @pytest.mark.asyncio
    async def test_concurrent_learning_cycles(self, advanced_learning_system):
        """Test handling of multiple concurrent learning cycles."""
        # Setup mock to simulate successful operations
        advanced_learning_system.model.fine_tune.return_value = {"status": "success", "loss": 0.1}
        advanced_learning_system.model.evaluate.return_value = {"accuracy": 0.85, "loss": 0.20}
        advanced_learning_system.feedback_collector.collect_feedback.return_value = [
            {"query": "test", "response": "resp", "rating": 5, "timestamp": datetime.now()}
        ]
        
        # First cycle should succeed
        result1 = await advanced_learning_system.run_continuous_learning_cycle()
        assert result1["status"] == "success"
        
        # Concurrent cycle should work after first completes
        result2 = await advanced_learning_system.run_continuous_learning_cycle()
        assert result2["status"] == "success"

    def test_configuration_validation_with_complex_configs(self, advanced_learning_system):
        """Test configuration validation with complex configuration objects."""
        complex_configs = [
            {
                "learning_rate": 0.001,
                "batch_size": 32,
                "max_epochs": 10,
                "optimizer": "adam",
                "scheduler": "cosine",
                "warmup_steps": 100
            },
            {
                "learning_rate": 0.01,
                "batch_size": 16,
                "max_epochs": 20,
                "dropout_rate": 0.1,
                "weight_decay": 0.01
            },
            {
                "learning_rate": 0.005,
                "batch_size": 64,
                "max_epochs": 15,
                "gradient_clipping": 1.0,
                "early_stopping_patience": 3
            }
        ]
        
        for config in complex_configs:
            result = advanced_learning_system.validate_configuration(config)
            assert result is True

    def test_training_data_preprocessing_edge_cases(self, advanced_learning_system):
        """Test training data preprocessing with edge cases."""
        edge_case_data = [
            {"input": "   Leading and trailing spaces   ", "output": "   Spaced output   "},
            {"input": "Multiple\n\nNewlines\n\n", "output": "Newline\nHandling"},
            {"input": "Tab\tSeparated\tValues", "output": "Tab\tOutput"},
            {"input": "Mixed whitespace \t\n ", "output": "Clean output"},
        ]
        
        # Should handle edge cases without errors
        result = advanced_learning_system.validate_training_data(edge_case_data)
        assert result is True

    def test_feedback_analytics_and_insights(self, advanced_learning_system):
        """Test feedback analytics and insight generation."""
        # Create comprehensive feedback dataset
        feedback_data = []
        for i in range(100):
            rating = (i % 5) + 1  # Cycle through ratings 1-5
            feedback_data.append({
                "query": f"Query {i}",
                "response": f"Response {i}",
                "rating": rating,
                "timestamp": datetime.now() - timedelta(days=i % 30),
                "category": f"category_{i % 3}",
                "user_type": f"user_type_{i % 2}"
            })
        
        # Test various filtering and analysis scenarios
        high_quality = advanced_learning_system.filter_high_quality_feedback(feedback_data, min_rating=4)
        medium_quality = advanced_learning_system.filter_high_quality_feedback(feedback_data, min_rating=3)
        
        assert len(high_quality) == 40  # Ratings 4 and 5
        assert len(medium_quality) == 60  # Ratings 3, 4, and 5

    @pytest.mark.asyncio
    async def test_learning_system_state_persistence(self, advanced_learning_system):
        """Test learning system state persistence across operations."""
        initial_state = {
            "version": advanced_learning_system.model_version,
            "training_samples": advanced_learning_system.total_training_samples,
            "feedback_samples": advanced_learning_system.total_feedback_samples,
            "errors": advanced_learning_system.error_count
        }
        
        # Perform operations that should modify state
        advanced_learning_system.model.fine_tune.return_value = {"status": "success", "loss": 0.1}
        advanced_learning_system.data_loader.load_training_data.return_value = [
            {"input": "test", "output": "test"}
        ]
        
        await advanced_learning_system.fine_tune_model()
        
        # Verify state changes
        assert advanced_learning_system.model_version == initial_state["version"] + 1
        assert advanced_learning_system.total_training_samples > initial_state["training_samples"]
        assert advanced_learning_system.last_training_time is not None


class TestLLMContinuousLearningSystemRobustness:
    """Test suite for system robustness and fault tolerance."""

    @pytest.fixture
    def resilient_learning_system(self):
        """Create a learning system configured for resilience testing."""
        mock_model = Mock()
        mock_data_loader = Mock()
        mock_feedback_collector = Mock()
        
        return LLMContinuousLearningSystem(
            model=mock_model,
            data_loader=mock_data_loader,
            feedback_collector=mock_feedback_collector,
            learning_rate=0.001,
            batch_size=16,
            max_epochs=5
        )

    def test_system_recovery_after_multiple_failures(self, resilient_learning_system):
        """Test system recovery after multiple consecutive failures."""
        # Simulate multiple evaluation failures
        resilient_learning_system.model.evaluate.side_effect = [
            Exception("First failure"),
            Exception("Second failure"), 
            Exception("Third failure"),
            {"accuracy": 0.85, "loss": 0.20}  # Finally succeeds
        ]
        
        initial_error_count = resilient_learning_system.error_count
        
        # First three attempts should fail
        for i in range(3):
            with pytest.raises(Exception):
                resilient_learning_system.evaluate_model_performance()
        
        # Fourth attempt should succeed
        result = resilient_learning_system.evaluate_model_performance()
        assert result["accuracy"] == 0.85
        assert resilient_learning_system.error_count == initial_error_count + 3

    @pytest.mark.asyncio
    async def test_training_interruption_and_resume(self, resilient_learning_system):
        """Test training interruption handling and resume capability."""
        # Simulate training interruption
        resilient_learning_system.model.fine_tune.side_effect = [
            KeyboardInterrupt("Training interrupted"),
        ]
        
        with pytest.raises(KeyboardInterrupt, match="Training interrupted"):
            await resilient_learning_system.fine_tune_model()
        
        # Verify training state is properly reset
        assert not resilient_learning_system._is_training
        
        # Should be able to start training again
        resilient_learning_system.model.fine_tune.side_effect = None
        resilient_learning_system.model.fine_tune.return_value = {"status": "success", "loss": 0.1}
        
        result = await resilient_learning_system.fine_tune_model()
        assert result["status"] == "success"

    def test_data_corruption_detection_and_handling(self, resilient_learning_system):
        """Test detection and handling of corrupted training data."""
        corrupted_data_scenarios = [
            # Malformed data structures
            [{"input": "valid", "invalid_key": "invalid"}],
            [{"output": "missing input key"}],
            [{"input": "missing output key"}],
            # Type errors
            [{"input": 123, "output": "should be string"}],
            [{"input": "valid", "output": ["should", "be", "string"]}],
            # Nested corruption
            [{"input": {"nested": "invalid"}, "output": "string"}],
        ]
        
        for corrupted_data in corrupted_data_scenarios:
            with pytest.raises(ValueError):
                resilient_learning_system.validate_training_data(corrupted_data)

    def test_extreme_memory_conditions(self, resilient_learning_system):
        """Test behavior under extreme memory conditions."""
        # Simulate very large data processing
        large_dataset = [
            {"input": "x" * 1000, "output": "y" * 1000} 
            for _ in range(1000)
        ]
        
        resilient_learning_system.data_loader.load_training_data.return_value = large_dataset
        resilient_learning_system.batch_size = 1  # Force many small batches
        
        # Should handle large datasets without errors
        batches = resilient_learning_system.create_training_batches()
        assert len(batches) == 1000
        assert all(len(batch) == 1 for batch in batches)

    @pytest.mark.parametrize("network_error", [
        ConnectionError("Network unreachable"),
        TimeoutError("Request timeout"), 
        OSError("Connection reset"),
        RuntimeError("Service unavailable")
    ])
    def test_network_failure_resilience(self, resilient_learning_system, network_error):
        """Test resilience to various network failures."""
        resilient_learning_system.feedback_collector.collect_feedback.side_effect = network_error
        
        # Should handle network errors gracefully
        with pytest.raises(type(network_error)):
            resilient_learning_system.collect_feedback()

    def test_filesystem_permission_handling(self, resilient_learning_system):
        """Test handling of filesystem permission issues."""
        # Simulate permission denied for checkpoint operations
        resilient_learning_system.model.save_checkpoint.side_effect = PermissionError("Permission denied")
        
        with pytest.raises(PermissionError, match="Permission denied"):
            resilient_learning_system.save_model_checkpoint("/restricted/path/checkpoint.pkl")

    def test_resource_exhaustion_scenarios(self, resilient_learning_system):
        """Test behavior under resource exhaustion scenarios."""
        # Simulate out of memory conditions
        resilient_learning_system.model.fine_tune = AsyncMock(
            side_effect=MemoryError("Out of memory")
        )
        
        with pytest.raises(MemoryError, match="Out of memory"):
            asyncio.run(resilient_learning_system.fine_tune_model())

    def test_concurrent_access_data_consistency(self, resilient_learning_system):
        """Test data consistency under concurrent access."""
        import threading
        import time
        
        results = []
        errors = []
        
        def concurrent_stats_reader():
            try:
                for _ in range(20):
                    stats = resilient_learning_system.get_system_statistics()
                    results.append(stats)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        def concurrent_stats_modifier():
            try:
                for i in range(20):
                    resilient_learning_system.total_training_samples += 1
                    resilient_learning_system.model_version += 1
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        # Run concurrent operations
        threads = [
            threading.Thread(target=concurrent_stats_reader),
            threading.Thread(target=concurrent_stats_modifier),
            threading.Thread(target=concurrent_stats_reader)
        ]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # Verify no errors occurred and results are reasonable
        assert len(errors) == 0
        assert len(results) == 40  # 2 reader threads * 20 calls each
        
        # All results should be valid dictionaries
        for result in results:
            assert isinstance(result, dict)
            assert "model_version" in result
            assert "total_training_samples" in result


class TestLLMContinuousLearningSystemPerformanceOptimization:
    """Test suite focused on performance optimization scenarios."""

    @pytest.fixture
    def performance_learning_system(self):
        """Create a learning system optimized for performance testing."""
        mock_model = Mock()
        mock_model.fine_tune = AsyncMock(return_value={"status": "success", "loss": 0.1})
        mock_model.evaluate = Mock(return_value={"accuracy": 0.85, "precision": 0.82})
        
        mock_data_loader = Mock()
        mock_feedback_collector = Mock()
        
        return LLMContinuousLearningSystem(
            model=mock_model,
            data_loader=mock_data_loader, 
            feedback_collector=mock_feedback_collector,
            learning_rate=0.001,
            batch_size=128,  # Larger batch size for performance
            max_epochs=10
        )

    def test_batch_size_optimization_impact(self, performance_learning_system):
        """Test impact of different batch sizes on processing efficiency."""
        test_data = create_sample_training_data(1000)
        performance_learning_system.data_loader.load_training_data.return_value = test_data
        
        batch_sizes = [1, 8, 16, 32, 64, 128, 256]
        batch_counts = []
        
        for batch_size in batch_sizes:
            performance_learning_system.batch_size = batch_size
            batches = performance_learning_system.create_training_batches()
            batch_counts.append(len(batches))
        
        # Verify inverse relationship between batch size and batch count
        assert batch_counts == [1000, 125, 63, 32, 16, 8, 4]
        
        # Verify total samples remain consistent
        for i, batch_size in enumerate(batch_sizes):
            performance_learning_system.batch_size = batch_size
            batches = performance_learning_system.create_training_batches()
            total_samples = sum(len(batch) for batch in batches)
            assert total_samples == 1000

    def test_large_feedback_dataset_processing(self, performance_learning_system):
        """Test processing of large feedback datasets."""
        # Create large feedback dataset
        large_feedback = create_sample_feedback_data(10000, rating_range=(1, 5))
        
        # Test filtering performance with different thresholds
        for min_rating in [1, 2, 3, 4, 5]:
            filtered = performance_learning_system.filter_high_quality_feedback(
                large_feedback, min_rating=min_rating
            )
            # Verify filtering works correctly
            assert all(item["rating"] >= min_rating for item in filtered)

    def test_memory_efficient_batch_processing(self, performance_learning_system):
        """Test memory-efficient processing of large batches."""
        # Create large dataset that would challenge memory
        large_dataset = []
        for i in range(5000):
            large_dataset.append({
                "input": f"Large input data sample {i} with extended content " * 10,
                "output": f"Large output data sample {i} with extended content " * 10
            })
        
        performance_learning_system.data_loader.load_training_data.return_value = large_dataset
        performance_learning_system.batch_size = 100
        
        batches = performance_learning_system.create_training_batches()
        
        # Verify efficient batching
        assert len(batches) == 50  # 5000 / 100
        assert all(len(batch) == 100 for batch in batches)

    @pytest.mark.asyncio
    async def test_async_operations_efficiency(self, performance_learning_system):
        """Test efficiency of async operations."""
        import time
        
        # Setup mock with realistic delays
        async def mock_fine_tune_with_delay(*args, **kwargs):
            await asyncio.sleep(0.01)  # Simulate processing time
            return {"status": "success", "loss": 0.1, "accuracy": 0.85}
        
        performance_learning_system.model.fine_tune = mock_fine_tune_with_delay
        
        # Measure async operation performance
        start_time = time.time()
        result = await performance_learning_system.fine_tune_model()
        end_time = time.time()
        
        assert result["status"] == "success"
        # Operation should complete reasonably quickly
        assert (end_time - start_time) < 1.0  # Less than 1 second

    def test_statistics_caching_and_efficiency(self, performance_learning_system):
        """Test statistics retrieval efficiency and potential caching."""
        # Call statistics multiple times
        stats_calls = []
        for _ in range(100):
            stats = performance_learning_system.get_system_statistics()
            stats_calls.append(stats)
        
        # Verify all calls return consistent data structure
        for stats in stats_calls:
            assert isinstance(stats, dict)
            assert len(stats) >= 5  # Should have multiple statistics
        
        # Verify consistency across calls
        first_stats = stats_calls[0]
        for stats in stats_calls[1:]:
            assert stats.keys() == first_stats.keys()


# Additional utility functions for comprehensive testing
def create_complex_training_data(size: int, complexity_level: str = "medium") -> List[Dict[str, str]]:
    """Create complex training data with varying characteristics."""
    complexity_configs = {
        "simple": {"input_words": 5, "output_words": 5},
        "medium": {"input_words": 20, "output_words": 15},
        "complex": {"input_words": 50, "output_words": 30}
    }
    
    config = complexity_configs.get(complexity_level, complexity_configs["medium"])
    
    data = []
    for i in range(size):
        input_text = " ".join([f"input_word_{j}_{i}" for j in range(config["input_words"])])
        output_text = " ".join([f"output_word_{j}_{i}" for j in range(config["output_words"])])
        
        data.append({
            "input": input_text,
            "output": output_text,
            "complexity": complexity_level,
            "sample_id": i
        })
    
    return data


def create_feedback_with_patterns(size: int, pattern: str = "realistic") -> List[Dict[str, Any]]:
    """Create feedback data with realistic rating patterns."""
    patterns = {
        "realistic": [5, 5, 4, 4, 4, 3, 3, 2, 1],  # Weighted toward higher ratings
        "uniform": [1, 2, 3, 4, 5] * 2,  # Uniform distribution
        "pessimistic": [1, 1, 2, 2, 3, 3, 4, 5],  # Weighted toward lower ratings
        "optimistic": [3, 4, 4, 5, 5, 5, 5, 5]  # Weighted toward higher ratings
    }
    
    rating_pattern = patterns.get(pattern, patterns["realistic"])
    
    feedback = []
    for i in range(size):
        rating = rating_pattern[i % len(rating_pattern)]
        feedback.append({
            "query": f"Query {i} with {pattern} pattern",
            "response": f"Response {i} for {pattern} feedback",
            "rating": rating,
            "timestamp": datetime.now() - timedelta(hours=i % 24),
            "pattern": pattern,
            "feedback_id": i
        })
    
    return feedback


# Performance benchmarking utilities
class PerformanceBenchmark:
    """Utility class for performance benchmarking in tests."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
    
    @property
    def elapsed_time(self):
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


# Additional pytest markers for comprehensive test organization
pytestmark.extend([
    pytest.mark.comprehensive,  # Mark for comprehensive testing
    pytest.mark.robustness,     # Mark for robustness testing
])
