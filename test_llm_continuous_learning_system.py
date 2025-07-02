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
from typing import List, Dict, Any, Tuple


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
    def mock_model_with_failures(self):
        """Create a mock model that can simulate various failure modes."""
        mock = Mock()
        mock.fine_tune = AsyncMock()
        mock.evaluate = Mock()
        mock.save_checkpoint = Mock()
        mock.load_checkpoint = Mock()
        return mock

    @pytest.fixture
    def mock_unreliable_data_loader(self):
        """Create a mock data loader that simulates unreliable behavior."""
        mock = Mock()
        mock.load_training_data = Mock()
        return mock

    @pytest.fixture
    def mock_intermittent_feedback_collector(self):
        """Create a mock feedback collector with intermittent failures."""
        mock = Mock()
        mock.collect_feedback = Mock()
        return mock

    @pytest.fixture
    def learning_system_advanced(self, mock_model_with_failures, mock_unreliable_data_loader, mock_intermittent_feedback_collector):
        """Create a learning system with failure-prone components."""
        return LLMContinuousLearningSystem(
            model=mock_model_with_failures,
            data_loader=mock_unreliable_data_loader,
            feedback_collector=mock_intermittent_feedback_collector
        )

    @pytest.mark.parametrize("learning_rate,batch_size,max_epochs,expected_error", [
        (-1.0, 16, 10, "Learning rate must be positive"),
        (0.001, -5, 10, "Batch size must be positive"),
        (0.001, 16, -1, "Max epochs must be positive"),
        (float('inf'), 16, 10, "Learning rate must be finite"),
        (0.001, float('inf'), 10, "Batch size must be finite"),
        (0.001, 16, float('inf'), "Max epochs must be finite"),
        (float('nan'), 16, 10, "Learning rate cannot be NaN"),
    ])
    def test_initialization_parameter_validation_comprehensive(self, mock_model_with_failures, 
                                                             mock_unreliable_data_loader, 
                                                             mock_intermittent_feedback_collector,
                                                             learning_rate, batch_size, max_epochs, expected_error):
        """Test comprehensive parameter validation during initialization."""
        with pytest.raises(ValueError, match=expected_error):
            LLMContinuousLearningSystem(
                model=mock_model_with_failures,
                data_loader=mock_unreliable_data_loader,
                feedback_collector=mock_intermittent_feedback_collector,
                learning_rate=learning_rate,
                batch_size=batch_size,
                max_epochs=max_epochs
            )

    @pytest.mark.asyncio
    async def test_cascading_failure_recovery(self, learning_system_advanced):
        """Test system behavior during cascading failures."""
        # Simulate multiple failure points
        learning_system_advanced.data_loader.load_training_data.side_effect = Exception("Data loading failed")
        learning_system_advanced.model.fine_tune.side_effect = Exception("Model training failed")
        learning_system_advanced.feedback_collector.collect_feedback.side_effect = Exception("Feedback collection failed")
        
        # Test that system handles cascading failures gracefully
        with pytest.raises(Exception):
            await learning_system_advanced.run_continuous_learning_cycle()
        
        # Verify error counting is accurate
        assert learning_system_advanced.error_count > 0

    @pytest.mark.parametrize("data_corruption_type", [
        "missing_keys",
        "wrong_types",
        "malformed_json",
        "encoding_issues",
        "circular_references"
    ])
    def test_data_corruption_handling(self, learning_system_advanced, data_corruption_type):
        """Test handling of various data corruption scenarios."""
        if data_corruption_type == "missing_keys":
            corrupted_data = [{"input": "test"}]  # Missing output
        elif data_corruption_type == "wrong_types":
            corrupted_data = [{"input": 123, "output": ["not", "a", "string"]}]
        elif data_corruption_type == "malformed_json":
            corrupted_data = ["not a dict"]
        elif data_corruption_type == "encoding_issues":
            corrupted_data = [{"input": "\x00\x01\x02", "output": "test"}]
        elif data_corruption_type == "circular_references":
            circular_dict = {"input": "test", "output": "test"}
            circular_dict["self"] = circular_dict
            corrupted_data = [circular_dict]
        
        with pytest.raises(ValueError):
            learning_system_advanced.validate_training_data(corrupted_data)

    @pytest.mark.asyncio
    async def test_resource_exhaustion_scenarios(self, learning_system_advanced):
        """Test behavior under resource exhaustion conditions."""
        # Simulate memory exhaustion
        learning_system_advanced.model.fine_tune.side_effect = MemoryError("Out of memory")
        
        with pytest.raises(MemoryError):
            await learning_system_advanced.fine_tune_model()
        
        # Verify system state is properly cleaned up
        assert not learning_system_advanced._is_training

    def test_extreme_data_sizes(self, learning_system_advanced):
        """Test handling of extremely large and small datasets."""
        # Test with extremely large dataset
        huge_data = [{"input": f"input_{i}", "output": f"output_{i}"} for i in range(100000)]
        learning_system_advanced.data_loader.load_training_data.return_value = huge_data
        learning_system_advanced.batch_size = 1000
        
        batches = learning_system_advanced.create_training_batches()
        assert len(batches) == 100  # 100000 / 1000
        
        # Test with single item dataset
        tiny_data = [{"input": "single", "output": "item"}]
        learning_system_advanced.data_loader.load_training_data.return_value = tiny_data
        learning_system_advanced.batch_size = 1000
        
        batches = learning_system_advanced.create_training_batches()
        assert len(batches) == 1
        assert len(batches[0]) == 1

    @pytest.mark.parametrize("rating_distribution", [
        [1] * 100,  # All minimum ratings
        [5] * 100,  # All maximum ratings
        list(range(1, 6)) * 20,  # Uniform distribution
        [1] * 80 + [5] * 20,  # Bimodal distribution
        [3] * 100,  # All neutral ratings
    ])
    def test_feedback_rating_distributions(self, learning_system_advanced, rating_distribution):
        """Test handling of various feedback rating distributions."""
        feedback_data = [
            {"query": f"query_{i}", "response": f"response_{i}", "rating": rating, "timestamp": datetime.now()}
            for i, rating in enumerate(rating_distribution)
        ]
        
        high_quality = learning_system_advanced.filter_high_quality_feedback(feedback_data, min_rating=4)
        expected_count = sum(1 for r in rating_distribution if r >= 4)
        assert len(high_quality) == expected_count

    @pytest.mark.asyncio
    async def test_training_interruption_and_resume(self, learning_system_advanced):
        """Test training interruption and resume capabilities."""
        # Set up a long-running training simulation
        async def slow_training():
            await asyncio.sleep(0.1)  # Simulate training time
            return {"status": "success", "loss": 0.1}
        
        learning_system_advanced.model.fine_tune = AsyncMock(side_effect=slow_training)
        
        # Start training
        training_task = asyncio.create_task(learning_system_advanced.fine_tune_model())
        
        # Wait briefly then check training state
        await asyncio.sleep(0.05)
        assert learning_system_advanced._is_training
        
        # Wait for completion
        result = await training_task
        assert result["status"] == "success"
        assert not learning_system_advanced._is_training

    def test_configuration_boundary_values(self, learning_system_advanced):
        """Test configuration validation with boundary values."""
        boundary_configs = [
            {"learning_rate": 1e-10, "batch_size": 1, "max_epochs": 1},  # Minimum values
            {"learning_rate": 1.0, "batch_size": 10000, "max_epochs": 1000},  # Large values
            {"learning_rate": 0.5, "batch_size": 2**10, "max_epochs": 2**8},  # Power of 2 values
        ]
        
        for config in boundary_configs:
            result = learning_system_advanced.validate_configuration(config)
            assert result is True

    @pytest.mark.parametrize("checkpoint_scenario", [
        "valid_checkpoint",
        "corrupted_checkpoint",
        "incompatible_version",
        "permission_denied",
        "disk_full"
    ])
    def test_checkpoint_error_scenarios(self, learning_system_advanced, checkpoint_scenario):
        """Test various checkpoint operation error scenarios."""
        checkpoint_path = "/tmp/test_checkpoint.pkl"
        
        if checkpoint_scenario == "valid_checkpoint":
            learning_system_advanced.save_model_checkpoint(checkpoint_path)
            learning_system_advanced.model.save_checkpoint.assert_called_once()
        elif checkpoint_scenario == "corrupted_checkpoint":
            learning_system_advanced.model.save_checkpoint.side_effect = Exception("Checkpoint corrupted")
            with pytest.raises(Exception, match="Checkpoint corrupted"):
                learning_system_advanced.save_model_checkpoint(checkpoint_path)
        elif checkpoint_scenario == "incompatible_version":
            learning_system_advanced.model.load_checkpoint.side_effect = ValueError("Incompatible checkpoint version")
            with pytest.raises(ValueError, match="Incompatible checkpoint version"):
                # Create a dummy file first
                with open(checkpoint_path, 'w') as f:
                    f.write("dummy")
                learning_system_advanced.load_model_checkpoint(checkpoint_path)
                os.unlink(checkpoint_path)
        elif checkpoint_scenario == "permission_denied":
            learning_system_advanced.model.save_checkpoint.side_effect = PermissionError("Permission denied")
            with pytest.raises(PermissionError):
                learning_system_advanced.save_model_checkpoint("/root/no_permission.pkl")
        elif checkpoint_scenario == "disk_full":
            learning_system_advanced.model.save_checkpoint.side_effect = OSError("No space left on device")
            with pytest.raises(OSError, match="No space left on device"):
                learning_system_advanced.save_model_checkpoint(checkpoint_path)

    def test_statistics_consistency_under_load(self, learning_system_advanced):
        """Test statistics consistency under concurrent access."""
        def heavy_operations():
            for _ in range(50):
                learning_system_advanced.total_training_samples += 1
                learning_system_advanced.total_feedback_samples += 2
                learning_system_advanced.error_count += 1
                stats = learning_system_advanced.get_system_statistics()
                # Verify statistics are internally consistent
                assert stats["total_training_samples"] >= 0
                assert stats["total_feedback_samples"] >= 0
                assert stats["error_count"] >= 0
        
        threads = [threading.Thread(target=heavy_operations) for _ in range(5)]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # Final consistency check
        final_stats = learning_system_advanced.get_system_statistics()
        assert final_stats["total_training_samples"] <= 250  # 5 threads * 50 operations
        assert final_stats["total_feedback_samples"] <= 500  # 5 threads * 50 * 2
        assert final_stats["error_count"] <= 250  # 5 threads * 50 operations

    @pytest.mark.asyncio
    async def test_async_operation_cancellation(self, learning_system_advanced):
        """Test proper handling of async operation cancellation."""
        # Create a cancellable training operation
        async def cancellable_training():
            try:
                await asyncio.sleep(1.0)  # Long operation
                return {"status": "success"}
            except asyncio.CancelledError:
                raise
        
        learning_system_advanced.model.fine_tune = AsyncMock(side_effect=cancellable_training)
        
        # Start training and cancel it
        training_task = asyncio.create_task(learning_system_advanced.fine_tune_model())
        await asyncio.sleep(0.1)  # Let training start
        training_task.cancel()
        
        with pytest.raises(asyncio.CancelledError):
            await training_task
        
        # Verify training flag is properly reset
        assert not learning_system_advanced._is_training

    def test_memory_leak_detection(self, learning_system_advanced):
        """Test for potential memory leaks during repeated operations."""
        initial_memory = learning_system_advanced.get_memory_usage()
        
        # Perform many operations that could cause memory leaks
        for _ in range(100):
            learning_system_advanced.data_loader.load_training_data.return_value = [
                {"input": f"test_{i}", "output": f"output_{i}"} for i in range(10)
            ]
            batches = learning_system_advanced.create_training_batches()
            learning_system_advanced.validate_training_data(learning_system_advanced.data_loader.load_training_data())
            learning_system_advanced.get_system_statistics()
        
        # Clean up and check memory
        learning_system_advanced.cleanup_memory()
        final_memory = learning_system_advanced.get_memory_usage()
        
        # Memory should not have grown excessively
        memory_growth = final_memory - initial_memory
        assert memory_growth < initial_memory * 2  # Less than 200% growth


class TestLLMContinuousLearningSystemStateTransitions:
    """Test suite for system state transitions and lifecycle management."""

    @pytest.fixture
    def mock_components(self):
        """Create mock components for state transition testing."""
        model = Mock()
        model.fine_tune = AsyncMock(return_value={"status": "success", "loss": 0.1})
        model.evaluate = Mock(return_value={"accuracy": 0.85})
        
        data_loader = Mock()
        data_loader.load_training_data = Mock(return_value=[
            {"input": "test", "output": "test"}
        ])
        
        feedback_collector = Mock()
        feedback_collector.collect_feedback = Mock(return_value=[
            {"query": "test", "response": "test", "rating": 5, "timestamp": datetime.now()}
        ])
        
        return model, data_loader, feedback_collector

    @pytest.fixture
    def learning_system_states(self, mock_components):
        """Create learning system for state testing."""
        model, data_loader, feedback_collector = mock_components
        return LLMContinuousLearningSystem(
            model=model,
            data_loader=data_loader,
            feedback_collector=feedback_collector
        )

    def test_initial_state_verification(self, learning_system_states):
        """Test that system starts in correct initial state."""
        stats = learning_system_states.get_system_statistics()
        
        assert stats["total_training_samples"] == 0
        assert stats["total_feedback_samples"] == 0
        assert stats["model_version"] == 1
        assert stats["error_count"] == 0
        assert stats["last_training_time"] is None
        assert stats["is_training"] is False

    @pytest.mark.asyncio
    async def test_training_state_transitions(self, learning_system_states):
        """Test state transitions during training operations."""
        # Initial state
        assert not learning_system_states._is_training
        
        # Create a training task that we can monitor
        async def monitored_training():
            # Check state immediately when training starts
            assert learning_system_states._is_training
            await asyncio.sleep(0.01)  # Simulate training work
            return {"status": "success", "loss": 0.1}
        
        learning_system_states.model.fine_tune.side_effect = monitored_training
        
        # Execute training
        result = await learning_system_states.fine_tune_model()
        
        # Verify final state
        assert not learning_system_states._is_training
        assert result["status"] == "success"
        assert learning_system_states.model_version == 2  # Should increment

    @pytest.mark.parametrize("operation_sequence", [
        ["train", "evaluate", "feedback"],
        ["feedback", "train", "evaluate"],
        ["evaluate", "feedback", "train"],
        ["train", "train", "evaluate"],  # Duplicate training should fail
    ])
    @pytest.mark.asyncio
    async def test_operation_sequence_states(self, learning_system_states, operation_sequence):
        """Test state consistency across different operation sequences."""
        for i, operation in enumerate(operation_sequence):
            if operation == "train":
                if i > 0 and operation_sequence[i-1] == "train":
                    # Second consecutive training should fail
                    learning_system_states._is_training = True
                    with pytest.raises(RuntimeError, match="Training already in progress"):
                        await learning_system_states.fine_tune_model()
                    learning_system_states._is_training = False
                else:
                    await learning_system_states.fine_tune_model()
            elif operation == "evaluate":
                learning_system_states.evaluate_model_performance()
            elif operation == "feedback":
                learning_system_states.collect_feedback()
        
        # Verify final state is consistent
        stats = learning_system_states.get_system_statistics()
        assert not stats["is_training"]

    def test_error_state_recovery(self, learning_system_states):
        """Test system recovery from error states."""
        # Introduce errors
        learning_system_states.model.evaluate.side_effect = Exception("Evaluation error")
        
        # Verify error increments
        initial_errors = learning_system_states.error_count
        try:
            learning_system_states.evaluate_model_performance()
        except Exception:
            pass
        
        assert learning_system_states.error_count == initial_errors + 1
        
        # Reset error condition and verify recovery
        learning_system_states.model.evaluate.side_effect = None
        learning_system_states.model.evaluate.return_value = {"accuracy": 0.9}
        
        result = learning_system_states.evaluate_model_performance()
        assert result["accuracy"] == 0.9

    def test_version_increment_tracking(self, learning_system_states):
        """Test proper version tracking across operations."""
        initial_version = learning_system_states.model_version
        
        # Simulate multiple training rounds
        for expected_version in range(initial_version + 1, initial_version + 5):
            asyncio.run(learning_system_states.fine_tune_model())
            assert learning_system_states.model_version == expected_version


class TestLLMContinuousLearningSystemAdvancedValidation:
    """Advanced validation tests for complex scenarios."""

    @pytest.fixture
    def validation_system(self):
        """Create system optimized for validation testing."""
        model = Mock()
        data_loader = Mock()
        feedback_collector = Mock()
        
        system = LLMContinuousLearningSystem(
            model=model,
            data_loader=data_loader,
            feedback_collector=feedback_collector
        )
        
        # Set validation constraints
        system.max_input_length = 1000
        system.max_output_length = 500
        
        return system

    @pytest.mark.parametrize("invalid_data,expected_error", [
        # Test various malformed data structures
        ([{"input": {"nested": "dict"}, "output": "test"}], "Invalid training data format"),
        ([{"input": ["list", "input"], "output": "test"}], "Invalid training data format"),
        ([{"input": "test", "output": {"nested": "dict"}}], "Invalid training data format"),
        ([{"input": "test", "output": ["list", "output"]}], "Invalid training data format"),
        # Test None and empty values
        ([{"input": None, "output": "test"}], "Empty inputs or outputs not allowed"),
        ([{"input": "test", "output": None}], "Empty inputs or outputs not allowed"),
        ([{"input": "", "output": "test"}], "Empty inputs or outputs not allowed"),
        ([{"input": "test", "output": ""}], "Empty inputs or outputs not allowed"),
        # Test whitespace-only values  
        ([{"input": "   ", "output": "test"}], "Empty inputs or outputs not allowed"),
        ([{"input": "test", "output": "   "}], "Empty inputs or outputs not allowed"),
        ([{"input": "\t\n", "output": "test"}], "Empty inputs or outputs not allowed"),
    ])
    def test_comprehensive_data_validation(self, validation_system, invalid_data, expected_error):
        """Test comprehensive data validation scenarios."""
        with pytest.raises(ValueError, match=expected_error):
            validation_system.validate_training_data(invalid_data)

    def test_input_length_validation_edge_cases(self, validation_system):
        """Test input length validation with edge cases."""
        # Test exact boundary
        boundary_input = "a" * validation_system.max_input_length
        valid_data = [{"input": boundary_input, "output": "test"}]
        assert validation_system.validate_training_data(valid_data) is True
        
        # Test exceeding boundary by one character
        exceeding_input = "a" * (validation_system.max_input_length + 1)
        invalid_data = [{"input": exceeding_input, "output": "test"}]
        with pytest.raises(ValueError, match="Input exceeds maximum length"):
            validation_system.validate_training_data(invalid_data)

    def test_output_length_validation_edge_cases(self, validation_system):
        """Test output length validation with edge cases."""
        # Test exact boundary
        boundary_output = "a" * validation_system.max_output_length
        valid_data = [{"input": "test", "output": boundary_output}]
        assert validation_system.validate_training_data(valid_data) is True
        
        # Test exceeding boundary by one character
        exceeding_output = "a" * (validation_system.max_output_length + 1)
        invalid_data = [{"input": "test", "output": exceeding_output}]
        with pytest.raises(ValueError, match="Output exceeds maximum length"):
            validation_system.validate_training_data(invalid_data)

    @pytest.mark.parametrize("special_chars", [
        "\x00\x01\x02\x03",  # Control characters
        "🚀🌟💫⭐",  # Emojis
        "αβγδεζηθ",  # Greek letters
        "中文测试",  # Chinese characters
        "🇺🇸🇬🇧🇫🇷",  # Flag emojis
        "♠♣♥♦",  # Card suits
        "∑∏∫∆∇",  # Mathematical symbols
        "©®™",  # Legal symbols
    ])
    def test_special_character_handling(self, validation_system, special_chars):
        """Test handling of various special characters."""
        data = [{"input": f"Test with {special_chars}", "output": f"Response with {special_chars}"}]
        # Should handle special characters gracefully
        assert validation_system.validate_training_data(data) is True

    def test_configuration_validation_edge_cases(self, validation_system):
        """Test configuration validation with edge cases."""
        # Test with extra keys
        config_with_extra = {
            "learning_rate": 0.01,
            "batch_size": 16,
            "max_epochs": 10,
            "extra_key": "should_be_ignored"
        }
        assert validation_system.validate_configuration(config_with_extra) is True
        
        # Test with string values (should fail)
        config_with_strings = {
            "learning_rate": "0.01",
            "batch_size": "16",
            "max_epochs": "10"
        }
        assert validation_system.validate_configuration(config_with_strings) is False


# Additional utility test functions
class TestLLMContinuousLearningSystemUtilities:
    """Test utility functions and helper methods."""

    def test_create_sample_training_data_function(self):
        """Test the utility function for creating sample training data."""
        sizes = [0, 1, 10, 100]
        for size in sizes:
            data = create_sample_training_data(size)
            assert len(data) == size
            if size > 0:
                assert all("input" in item and "output" in item for item in data)
                assert all(isinstance(item["input"], str) and isinstance(item["output"], str) for item in data)

    def test_create_sample_feedback_data_function(self):
        """Test the utility function for creating sample feedback data."""
        # Test default rating range
        data = create_sample_feedback_data(10)
        assert len(data) == 10
        assert all(1 <= item["rating"] <= 5 for item in data)
        
        # Test custom rating range
        data = create_sample_feedback_data(5, rating_range=(3, 7))
        assert len(data) == 5
        assert all(3 <= item["rating"] <= 7 for item in data)

    def test_utility_data_structure_consistency(self):
        """Test that utility functions create consistent data structures."""
        training_data = create_sample_training_data(5)
        feedback_data = create_sample_feedback_data(5)
        
        # Verify training data structure
        for item in training_data:
            assert isinstance(item, dict)
            assert set(item.keys()) == {"input", "output"}
        
        # Verify feedback data structure
        for item in feedback_data:
            assert isinstance(item, dict)
            assert set(item.keys()) == {"query", "response", "rating", "timestamp"}
            assert isinstance(item["timestamp"], datetime)


# Performance and stress tests
class TestLLMContinuousLearningSystemStress:
    """Stress tests for system reliability under extreme conditions."""

    @pytest.fixture
    def stress_test_system(self):
        """Create system for stress testing."""
        model = Mock()
        model.fine_tune = AsyncMock(return_value={"status": "success", "loss": 0.1})
        model.evaluate = Mock(return_value={"accuracy": 0.85})
        
        data_loader = Mock()
        feedback_collector = Mock()
        
        return LLMContinuousLearningSystem(
            model=model,
            data_loader=data_loader,
            feedback_collector=feedback_collector
        )

    @pytest.mark.stress
    def test_rapid_successive_operations(self, stress_test_system):
        """Test rapid successive operations for race conditions."""
        operations_count = 100
        
        # Rapid statistics access
        for _ in range(operations_count):
            stats = stress_test_system.get_system_statistics()
            assert isinstance(stats, dict)
        
        # Rapid configuration validation
        config = {"learning_rate": 0.01, "batch_size": 16, "max_epochs": 10}
        for _ in range(operations_count):
            result = stress_test_system.validate_configuration(config)
            assert result is True

    @pytest.mark.stress 
    def test_memory_pressure_simulation(self, stress_test_system):
        """Test system behavior under simulated memory pressure."""
        # Create large data structures repeatedly
        large_datasets = []
        for i in range(10):
            large_data = create_sample_training_data(1000)
            large_datasets.append(large_data)
            
            # Validate each dataset
            stress_test_system.data_loader.load_training_data.return_value = large_data
            batches = stress_test_system.create_training_batches()
            assert len(batches) > 0
        
        # Cleanup
        stress_test_system.cleanup_memory()

    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_concurrent_async_operations_stress(self, stress_test_system):
        """Test handling of many concurrent async operations."""
        # Create multiple async tasks that don't actually conflict
        async def non_training_async_op():
            await asyncio.sleep(0.001)
            return stress_test_system.get_system_statistics()
        
        # Run many concurrent non-training operations
        tasks = [non_training_async_op() for _ in range(50)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 50
        assert all(isinstance(result, dict) for result in results)


# Add markers for new test categories
pytestmark.extend([
    pytest.mark.comprehensive,  # Mark comprehensive test additions
    pytest.mark.advanced,       # Mark advanced scenario tests
])

# Additional pytest configuration
def pytest_configure_advanced(config):
    """Configure additional pytest markers for enhanced tests."""
    config.addinivalue_line("markers", "comprehensive: Comprehensive test coverage")
    config.addinivalue_line("markers", "advanced: Advanced scenario tests") 
    config.addinivalue_line("markers", "stress: Stress and load tests")
    config.addinivalue_line("markers", "validation: Data validation tests")
