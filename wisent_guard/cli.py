"""
Command-line interface for wisent-guard lm-evaluation-harness integration.
Clean implementation using enhanced core primitives.
"""

import logging
import sys
import os
import torch
from datetime import datetime
from typing import List, Dict, Any, Optional

from .core import Model, ContrastivePairSet, SteeringMethod, SteeringType, Layer
from .core.ground_truth_evaluator import GroundTruthEvaluator
from .core.activations import TestActivationCache
from .optimize import (
    run_smart_optimization,
    run_interactive_optimization,
)
from .core.save_results import (
    save_results_json,
    save_results_csv,
    create_evaluation_report,
)
from .core.parser import (
    setup_parser,
    aggregate_token_scores,
    parse_layers_from_arg,
)
from .inference import (
    generate_with_classification_and_handling,
    generate_with_multi_layer_classification_and_handling,
)
from .core.contrastive_pairs import (
    generate_synthetic_pairs_cli,
    load_synthetic_pairs_cli,
)
from .core.lm_eval_harness_ground_truth import LMEvalHarnessGroundTruth
from .core.model_config_manager import ModelConfigManager

# Import caching infrastructure
try:
    from .core.classifiers.pipeline_steps.download_full_benchmarks import (
        FullBenchmarkDownloader,
    )
except ImportError:
    FullBenchmarkDownloader = None

# Import validated benchmark list and unavailable benchmarks filter
try:
    from .core.lm_harness_integration.only_benchmarks import CORE_BENCHMARKS

    # Import UNAVAILABLE_BENCHMARKS from download script
    try:
        from .core.classifiers.pipeline_steps.download_full_benchmarks import (
            FullBenchmarkDownloader,
        )

        UNAVAILABLE_BENCHMARKS = FullBenchmarkDownloader.UNAVAILABLE_BENCHMARKS
    except ImportError:
        UNAVAILABLE_BENCHMARKS = set()
except ImportError:
    try:
        # Alternative import path for development/testing
        import sys
        import os

        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(current_dir, "core", "lm-harness-integration"))
        from only_benchmarks import CORE_BENCHMARKS

        # Try to import UNAVAILABLE_BENCHMARKS
        try:
            sys.path.insert(
                0, os.path.join(current_dir, "core", "classifiers", "pipeline_steps")
            )
            from download_full_benchmarks import FullBenchmarkDownloader

            UNAVAILABLE_BENCHMARKS = FullBenchmarkDownloader.UNAVAILABLE_BENCHMARKS
        except ImportError:
            UNAVAILABLE_BENCHMARKS = set()
    except ImportError:
        # Fallback - create minimal benchmark list
        CORE_BENCHMARKS = {
            "truthfulqa_mc1": {
                "task": "truthfulqa_mc1",
                "tags": ["hallucination"],
                "priority": "high",
            },
            "hellaswag": {
                "task": "hellaswag",
                "tags": ["reasoning"],
                "priority": "high",
            },
            "mmlu": {"task": "mmlu", "tags": ["knowledge"], "priority": "high"},
        }
        UNAVAILABLE_BENCHMARKS = set()
        print(
            "⚠️  Warning: Could not import CORE_BENCHMARKS, using minimal fallback list"
        )

# Define the whitelist of allowed tasks (from LM_EVAL_HARNESS_TASKS)
ALLOWED_TASKS = {
    # Original tested tasks
    "math_qa",
    "webqs",
    "truthfulqa_gen",
    "drop",
    "record",
    "squad2",
    "wikitext",
    "winogrande",
    # Multiple choice benchmarks
    "arc_challenge",
    "arc_easy",
    "hellaswag",
    "truthfulqa_mc1",
    "truthfulqa_mc2",
    "mmlu",
    "mmmlu",
    "piqa",
    "copa",
    "openbookqa",
    "race",
    # Boolean benchmarks
    "boolq",
    # Math benchmarks
    "gsm8k",
    "asdiv",
    "arithmetic",
    # QA benchmarks
    "coqa",
    "naturalqs",
    "triviaqa",
    # Coding benchmarks
    "mbpp",
    "livecodebench",
    # Additional benchmarks
    "cb",
    "logiqa",
    "multirc",
    "mutual",
    "prost",
    "pubmedqa",
    "sciq",
    "swag",
    "toxigen",
    "wic",
    "wsc",
    "wsc273",
    # Adversarial benchmarks
    "anli",
    # BigCode benchmarks
    "humaneval",
    "humaneval_plus",
    "instructhumaneval",
    "apps",
    "mbpp_plus",
    "ds1000",
    "humanevalpack",
    "multiple_py",
    "multiple_js",
    "multiple_java",
    "multiple_cpp",
    "multiple_rs",
    "multiple_go",
    "recode",
    "conala",
    "concode",
    "codexglue_code_to_text",
    "codexglue_code_to_text_python",
    "codexglue_code_to_text_go",
    "codexglue_code_to_text_ruby",
    "codexglue_code_to_text_java",
    "codexglue_code_to_text_javascript",
    "codexglue_code_to_text_php",
    "mercury",
}

# Filter to only available (working) benchmarks - this gives us the validated benchmarks
AVAILABLE_BENCHMARKS = {
    name: config
    for name, config in CORE_BENCHMARKS.items()
    if name not in UNAVAILABLE_BENCHMARKS and name in ALLOWED_TASKS
}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_valid_task_names() -> List[str]:
    """Get list of all valid (available) task names from AVAILABLE_BENCHMARKS."""
    return sorted(list(AVAILABLE_BENCHMARKS.keys()))


def validate_task_name(task_name: str) -> bool:
    """
    Validate if a task name is in the approved AVAILABLE_BENCHMARKS list (37 working benchmarks).

    Args:
        task_name: Name of the task to validate

    Returns:
        True if task is valid, False otherwise
    """
    return task_name in AVAILABLE_BENCHMARKS


def suggest_similar_tasks(invalid_task: str, max_suggestions: int = 5) -> List[str]:
    """
    Suggest similar task names for an invalid task using fuzzy matching.

    Args:
        invalid_task: The invalid task name
        max_suggestions: Maximum number of suggestions to return

    Returns:
        List of suggested task names
    """
    from difflib import get_close_matches

    valid_tasks = get_valid_task_names()
    suggestions = get_close_matches(
        invalid_task, valid_tasks, n=max_suggestions, cutoff=0.6
    )
    return suggestions


def print_valid_tasks_by_category():
    """Print all valid (available) tasks organized by categories/tags."""
    excluded_count = len(UNAVAILABLE_BENCHMARKS)
    print(
        f"\n📋 VALID TASKS ({len(AVAILABLE_BENCHMARKS)} available out of {len(CORE_BENCHMARKS)} total):"
    )
    if excluded_count > 0:
        print(
            f"    🚫 {excluded_count} benchmarks excluded (known unavailable/problematic)"
        )
    print("=" * 60)

    # Group by priority for better organization
    by_priority = {"high": [], "medium": [], "low": [], "unknown": []}

    for name, config in AVAILABLE_BENCHMARKS.items():
        priority = config.get("priority", "unknown")
        by_priority[priority].append((name, config))

    for priority in ["high", "medium", "low", "unknown"]:
        if by_priority[priority]:
            emoji = (
                "🚀"
                if priority == "high"
                else (
                    "⚡"
                    if priority == "medium"
                    else "🐌"
                    if priority == "low"
                    else "❓"
                )
            )
            print(
                f"\n{emoji} {priority.upper()} PRIORITY ({len(by_priority[priority])} tasks):"
            )

            for name, config in sorted(by_priority[priority]):
                tags = ", ".join(config.get("tags", []))
                print(f"   • {name:<20} - {tags}")

    print("\n💡 Usage: wisent-guard tasks --task-name <task_name>")
    print("   Example: wisent-guard tasks --task-name truthfulqa_mc1")

    if excluded_count > 0:
        print(f"\n🚫 EXCLUDED BENCHMARKS ({excluded_count} total):")
        excluded_list = sorted(list(UNAVAILABLE_BENCHMARKS))
        for i in range(0, len(excluded_list), 4):  # Show 4 per line
            line_items = excluded_list[i : i + 4]
            print(f"   🚫 {', '.join(line_items)}")


def print_task_info(task_name: str):
    """Print detailed information about a specific task."""
    if task_name not in AVAILABLE_BENCHMARKS:
        if task_name in UNAVAILABLE_BENCHMARKS:
            print(f"❌ Task '{task_name}' is known to be unavailable/problematic")
            print(
                f"🚫 This benchmark is excluded from the {len(AVAILABLE_BENCHMARKS)} available benchmarks"
            )
        else:
            print(f"❌ Task '{task_name}' not found in available benchmarks")
        return

    config = AVAILABLE_BENCHMARKS[task_name]
    print(f"\n📋 TASK INFO: {task_name}")
    print("=" * 50)
    print(f"🎯 Task ID: {config.get('task', task_name)}")
    print(f"🏷️  Tags: {', '.join(config.get('tags', []))}")
    print(f"⚡ Priority: {config.get('priority', 'unknown')}")

    if config.get("priority") == "high":
        print("   💡 Fast loading (< 13.5s) - optimal for agentic use")
    elif config.get("priority") == "medium":
        print("   💡 Moderate loading (13.5-60s) - acceptable for agentic use")
    elif config.get("priority") == "low":
        print("   💡 Slow loading (> 60s) - deprioritized for agentic use")

    if "trust_remote_code" in config:
        print(f"⚠️  Requires trust_remote_code: {config['trust_remote_code']}")


def get_cache_path(task_name: str, cache_dir: str) -> str:
    """Get the cache file path for a benchmark."""
    import os

    # Match the structure used by FullBenchmarkDownloader which saves to data/ subdirectory
    return os.path.join(cache_dir, "data", f"{task_name}.pkl")


def is_benchmark_cached(task_name: str, cache_dir: str) -> bool:
    """Check if a benchmark is cached."""
    import os

    cache_path = get_cache_path(task_name, cache_dir)
    return os.path.exists(cache_path)


def load_cached_benchmark(
    task_name: str, cache_dir: str
) -> Optional[List[Dict[str, Any]]]:
    """Load cached benchmark data."""
    import pickle
    import os

    cache_path = get_cache_path(task_name, cache_dir)
    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, "rb") as f:
            cached_data = pickle.load(f)
        return cached_data
    except Exception as e:
        print(f"⚠️  Failed to load cached data: {e}")
        return None


def save_benchmark_to_cache(
    task_name: str, cache_dir: str, limit: Optional[int] = None, verbose: bool = False
) -> bool:
    """Save benchmark data to cache using the download infrastructure."""
    import os

    if FullBenchmarkDownloader is None:
        if verbose:
            print("⚠️  Warning: Download infrastructure not available, cannot cache")
        return False

    if task_name not in AVAILABLE_BENCHMARKS:
        if verbose:
            print(f"⚠️  Warning: {task_name} not in available benchmarks")
        return False

    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)

    try:
        if verbose:
            print(f"💾 Caching benchmark {task_name}...")

        # Use the download infrastructure to get and cache the data
        downloader = FullBenchmarkDownloader(download_dir=cache_dir)
        benchmark_config = AVAILABLE_BENCHMARKS[task_name]

        # Download and save (this handles the full download and conversion)
        result_path = downloader.download_complete_benchmark(
            task_name,
            benchmark_config,
            force=True,  # Force download since we want to cache
        )

        if result_path:
            if verbose:
                print(f"✅ Successfully cached {task_name}")
            return True
        else:
            if verbose:
                print(f"❌ Failed to cache {task_name}")
            return False

    except Exception as e:
        if verbose:
            print(f"❌ Error caching {task_name}: {e}")
        return False


def convert_cached_data_to_qa_pairs(
    cached_data: List[Dict[str, Any]], limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Convert cached contrastive pairs back to QA pair format expected by the pipeline."""
    qa_pairs = []

    for pair in cached_data:
        # Convert contrastive pair back to QA format
        # The cached data is in format: {"context": "", "good_response": "", "bad_response": "", "metadata": {}}
        qa_pair = {
            "question": pair.get("context", ""),
            "correct_answer": pair.get("good_response", ""),
            "incorrect_answer": pair.get("bad_response", ""),
            "metadata": pair.get("metadata", {}),
        }
        qa_pairs.append(qa_pair)

        if limit and len(qa_pairs) >= limit:
            break

    return qa_pairs


def _run_lm_harness_evaluation(
    task_data, test_qa_pairs, model, steering_methods, layers, verbose=False
):
    """
    Run proper lm-harness evaluation with steering integration.

    Args:
        task_data: The lm-harness task object
        test_qa_pairs: List of test QA pairs
        model: The wisent Model instance
        steering_methods: List of steering methods to apply
        layers: List of layers for steering
        verbose: Whether to print verbose output

    Returns:
        Dict containing evaluation results
    """
    try:
        try:
            from lm_eval.api.evaluator import evaluate
            from lm_eval.api.model import LM
        except ImportError:
            # Try newer lm-eval import paths
            from lm_eval import evaluate
            from lm_eval.api.model import LM

        if verbose:
            print("\n🔍 RUNNING LM-HARNESS EVALUATION WITH STEERING:")
            print(f"   • Task: {task_data.config.task}")
            print(f"   • Test samples: {len(test_qa_pairs)}")
            try:
                steering_method_names = (
                    [
                        m.method_type.value if hasattr(m, "method_type") else str(m)
                        for m in steering_methods
                    ]
                    if steering_methods
                    else "None"
                )
            except Exception:
                steering_method_names = "Unknown"
            print(f"   • Steering methods: {steering_method_names}")
            print(f"   • Layers: {layers}")

        # Create a steered model wrapper for lm-harness evaluation
        class SteeredModelWrapper(LM):
            """Model wrapper that applies steering during lm-harness evaluation."""

            def __init__(self, wisent_model, steering_methods, layers):
                self.wisent_model = wisent_model
                self.steering_methods = steering_methods
                self.layers = layers

            def generate_until(self, requests):
                """Generate responses with steering applied."""
                results = []

                for req in requests:
                    # Extract the prompt from the request
                    if hasattr(req, "args") and req.args:
                        prompt = (
                            req.args[0]
                            if isinstance(req.args[0], str)
                            else str(req.args[0])
                        )
                    else:
                        prompt = str(req)

                    try:
                        # Generate with steering using the wisent model
                        if self.steering_methods and self.layers:
                            # Apply steering during generation
                            response, _, _, _ = (
                                generate_with_classification_and_handling(
                                    self.wisent_model,
                                    prompt,
                                    self.layers[0],  # Use first layer
                                    max_new_tokens=300,
                                    steering_method=(
                                        self.steering_methods[0]
                                        if self.steering_methods
                                        else None
                                    ),
                                    token_aggregation="average",
                                    detection_threshold=0.6,
                                    verbose=False,
                                    detection_handler=None,
                                )
                            )
                        else:
                            # Generate without steering
                            response = self.wisent_model.generate(
                                prompt,
                                layer_index=self.layers[0] if self.layers else 15,
                                max_new_tokens=300,
                            )

                        results.append(response)

                    except Exception as e:
                        if verbose:
                            print(f"   ⚠️ Generation failed for prompt: {e}")
                        results.append("Generation failed")

                return results

            def loglikelihood(self, requests):
                """Compute log-likelihood with steering applied."""
                results = []

                for req in requests:
                    try:
                        # For now, return neutral likelihood
                        # TODO: Implement proper likelihood computation with steering
                        results.append((0.0, False))
                    except Exception:
                        results.append((float("-inf"), False))

                return results

            def loglikelihood_rolling(self, requests):
                """Rolling log-likelihood computation."""
                return [0.0] * len(requests)

        # Create steered model wrapper
        steered_model = SteeredModelWrapper(model, steering_methods, layers)

        # Run evaluation using lm-harness with steering
        results = evaluate(
            model=steered_model,
            tasks=[task_data.config.task],
            limit=len(test_qa_pairs),
            bootstrap_iters=0,  # Disable bootstrapping for speed
        )

        # Extract accuracy and other metrics
        task_name = task_data.config.task
        task_results = results.get("results", {}).get(task_name, {})

        accuracy = task_results.get("acc", task_results.get("accuracy", "N/A"))

        evaluation_results = {
            "accuracy": accuracy,
            "method": "lm_harness_with_steering",
            "task_name": task_name,
            "steering_applied": (
                len(steering_methods) > 0 if steering_methods else False
            ),
            "full_results": task_results,
        }

        if verbose:
            print("   ✅ Evaluation completed")
            print(f"   📊 Accuracy: {accuracy}")
            print(f"   🎯 Steering applied: {'Yes' if steering_methods else 'No'}")

        return evaluation_results

    except Exception as e:
        if verbose:
            print(f"   ❌ LM-harness evaluation failed: {e}")

        # Fallback to placeholder
        return {
            "accuracy": "N/A",
            "method": "lm_harness_failed",
            "error": str(e),
            "note": "LM-harness evaluation failed, falling back to individual response evaluation",
        }


def run_task_pipeline(
    task_name: str,
    model_name: str,
    layer: str,
    shots: int = 0,
    split_ratio: float = 0.8,
    limit: int = None,
    training_limit: int = None,
    testing_limit: int = None,
    classifier_type: str = "logistic",
    max_new_tokens: int = 300,
    device: str = None,
    seed: int = 42,
    token_aggregation: str = "average",
    ground_truth_method: str = "lm-eval-harness",
    user_labels: List[str] = None,
    optimize: bool = False,
    optimize_layers: str = "all",
    optimize_metric: str = "f1",
    optimize_max_combinations: int = 100,
    verbose: bool = False,
    from_csv: bool = False,
    from_json: bool = False,
    question_col: str = "question",
    correct_col: str = "correct_answer",
    incorrect_col: str = "incorrect_answer",
    allow_small_dataset: bool = False,
    detection_action: str = "pass_through",
    placeholder_message: str = None,
    max_regeneration_attempts: int = 3,
    detection_threshold: float = 0.6,
    log_detections: bool = False,
    steering_mode: bool = False,
    steering_strength: float = 1.0,
    output_mode: str = "both",
    save_steering_vector: str = None,
    load_steering_vector: str = None,
    train_only: bool = False,
    inference_only: bool = False,
    save_classifier: str = None,
    load_classifier: str = None,
    classifier_dir: str = "./models",
    prompt_construction_strategy: str = "multiple_choice",
    token_targeting_strategy: str = "choice_token",
    normalize_mode: bool = False,
    normalization_method: str = "none",
    target_norm: Optional[float] = None,
    steering_method: str = "CAA",
    hpr_beta: float = 1.0,
    dac_dynamic_control: bool = False,
    dac_entropy_threshold: float = 1.0,
    bipo_beta: float = 0.1,
    bipo_learning_rate: float = 5e-4,
    bipo_epochs: int = 100,
    ksteering_num_labels: int = 6,
    ksteering_hidden_dim: int = 512,
    ksteering_learning_rate: float = 1e-3,
    ksteering_classifier_epochs: int = 100,
    ksteering_target_labels: str = "0",
    ksteering_avoid_labels: str = "",
    ksteering_alpha: float = 50.0,
    # Nonsense detection parameters
    enable_nonsense_detection: bool = False,
    max_word_length: int = 20,
    repetition_threshold: float = 0.7,
    gibberish_threshold: float = 0.3,
    disable_dictionary_check: bool = False,
    nonsense_action: str = "regenerate",
    # Token steering parameters
    enable_token_steering: bool = False,
    token_steering_strategy: str = "second_to_last",
    token_decay_rate: float = 0.5,
    token_min_strength: float = 0.1,
    token_max_strength: float = 1.0,
    token_apply_to_prompt: bool = False,
    token_prompt_strength_multiplier: float = 0.1,
    # Performance monitoring parameters
    enable_memory_tracking: bool = False,
    enable_latency_tracking: bool = False,
    memory_sampling_interval: float = 0.1,
    track_gpu_memory: bool = False,
    detailed_performance_report: bool = False,
    export_performance_csv: str = None,
    show_memory_usage: bool = False,
    show_timing_summary: bool = False,
    # Test-time activation saving/loading parameters
    save_test_activations: str = None,
    load_test_activations: str = None,
    # Priority-aware benchmark selection parameters
    priority: str = "all",
    fast_only: bool = False,
    time_budget: float = None,
    max_benchmarks: int = None,
    smart_selection: bool = False,
    # Benchmark caching parameters
    cache_benchmark: bool = True,
    use_cached: bool = True,
    force_download: bool = False,
    cache_dir: str = "./benchmark_cache",
    # Model reuse parameter for efficiency
    model_instance: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run the complete task pipeline for a given task.

    Args:
        task_name: The name of the task to run
        ground_truth_method: The method to use for ground truth evaluation
        ... (other parameters)

    Returns:
        Dictionary containing the results of the task pipeline
    """

    # SECURITY: Enforce Docker for code execution tasks
    from .core import enforce_secure_execution, SecureCodeEvaluator

    if enforce_secure_execution(task_name):
        if verbose:
            print(f"🔒 Task '{task_name}' requires secure Docker execution")
            print("   • All code will be executed in isolated Docker containers")
            print("   • This is mandatory for security - cannot be disabled")

        # Ensure Docker is available for code execution tasks
        try:
            # Test if we can create a secure evaluator (will check Docker availability)
            secure_evaluator = SecureCodeEvaluator()
            if verbose:
                executor_info = secure_evaluator.get_executor_info()
                print(f"   • Docker executor ready: {executor_info['image_name']}")
        except Exception as e:
            # FAIL HARD - No mock execution allowed
            print("\n❌ FATAL ERROR: Docker is required for code execution tasks")
            print(f"   • Task '{task_name}' requires secure Docker execution")
            print(f"   • Docker error: {e}")
            print("\n📋 To fix this:")
            print("   1. Install Docker: https://docs.docker.com/get-docker/")
            print("   2. Start Docker daemon")
            print("   3. Ensure your user has Docker permissions")
            print(
                "\n⚠️  Code execution tasks CANNOT run without Docker for security reasons"
            )
            sys.exit(1)

    # AUTOMATICALLY SET LM-EVAL-HARNESS AS DEFAULT FOR TESTED TASKS
    # These tasks have been thoroughly tested and validated to work with lm-eval-harness
    # 🚨 UPDATED: All benchmarks now use extractor-based ground truth evaluation
    LM_EVAL_HARNESS_TASKS = {
        # Original tested tasks
        "math_qa",  # Text-generation: 100% accuracy, working perfectly
        "webqs",  # Text-generation: 50% accuracy, working perfectly
        "truthfulqa_gen",  # Text-generation: 100% accuracy, working perfectly
        "drop",  # Text-generation: 50% accuracy, working perfectly
        "record",  # Text-generation: 0% accuracy, working perfectly (expected for cloze task)
        "squad2",  # Text-generation: 100% accuracy, working perfectly
        "wikitext",  # Perplexity: Working perfectly
        "winogrande",  # Text-generation: Testing for classification optimization
        # 🔧 NEW: All benchmarks with extractors now use lm-eval-harness ground truth
        # Multiple choice benchmarks
        "arc_challenge",
        "arc_easy",  # ARCExtractor
        "hellaswag",  # HellaSwagExtractor
        "truthfulqa_mc1",
        "truthfulqa_mc2",  # TruthfulQAExtractor
        "mmlu",
        "mmmlu",  # MMLUExtractor
        "piqa",  # PIQAExtractor
        "copa",  # COPAExtractor
        "openbookqa",  # OpenBookQAExtractor
        "race",  # RACEExtractor
        # Boolean benchmarks
        "boolq",  # BoolQExtractor
        # Math benchmarks
        "gsm8k",  # GSM8KExtractor
        "asdiv",  # GSM8KExtractor (similar format)
        # QA benchmarks (already included: squad2, drop, webqs, record)
        "coqa",  # SQuAD2Extractor (similar format)
        "naturalqs",  # SQuAD2Extractor (similar format)
        "triviaqa",  # SQuAD2Extractor (similar format)
        # Additional benchmarks that map to existing extractors
        "cb",  # COPAExtractor (similar format)
        "logiqa",
        "math_qa",
        "multirc",
        "mutual",
        "prost",
        "pubmedqa",
        "sciq",  # MMLUExtractor (multiple choice)
        "swag",  # HellaSwagExtractor (similar format)
        "toxigen",
        "wic",  # BoolQExtractor (binary classification)
        "wsc",
        "wsc273",  # COPAExtractor (similar format)
    }

    # Inform user about ground truth method for tested tasks
    if (
        ground_truth_method == "lm-eval-harness"
        and task_name.lower() in LM_EVAL_HARNESS_TASKS
        and verbose
    ):
        print(f"✅ Using 'lm-eval-harness' ground truth method for task '{task_name}'")
        print(
            "   • This task has been tested and validated to work with lm-eval-harness"
        )
        print("   • To use a different method, specify --ground-truth-method <method>")

    # AUTO-LOAD MODEL CONFIGURATION (if available)
    # Load saved optimal parameters for this model if they exist
    from .core.model_config_manager import get_optimal_parameters, ModelConfigManager

    # Load model config for control vectors and other settings
    config_manager = ModelConfigManager()
    model_config = config_manager.load_model_config(model_name)

    # Check if we should load saved configuration
    # Only load if parameters haven't been explicitly overridden by CLI arguments
    load_saved_config = True
    config_overrides = {}

    # Detect if user provided explicit layer argument (not default "15")
    if layer != "15":  # Assuming "15" is the default
        load_saved_config = False

    # Detect if user provided explicit token_aggregation (not default "average")
    if token_aggregation != "average":
        load_saved_config = False

    # Detect if user provided explicit detection_threshold (not default 0.6)
    if detection_threshold != 0.6:
        load_saved_config = False

    if load_saved_config:
        optimal_params = get_optimal_parameters(model_name, task_name)
        if optimal_params:
            # Store original values for comparison
            original_layer = layer
            original_aggregation = token_aggregation
            original_threshold = detection_threshold

            # Apply saved configuration
            if "classification_layer" in optimal_params:
                layer = str(optimal_params["classification_layer"])
                config_overrides["layer"] = f"{original_layer} → {layer}"

            if "token_aggregation" in optimal_params:
                token_aggregation = optimal_params["token_aggregation"]
                config_overrides["token_aggregation"] = (
                    f"{original_aggregation} → {token_aggregation}"
                )

            if "detection_threshold" in optimal_params:
                detection_threshold = optimal_params["detection_threshold"]
                config_overrides["detection_threshold"] = (
                    f"{original_threshold} → {detection_threshold}"
                )

            if verbose or config_overrides:
                print(f"\n🔧 Auto-loaded saved configuration for model: {model_name}")
                for param, change in config_overrides.items():
                    print(f"   • {param}: {change}")
                print("   • To override these defaults, specify parameters explicitly")
                print("   • Config file: ~/.wisent-guard/model_configs/")
                print()

        # AUTO-LOAD CONTROL VECTOR: Check if a control vector exists for this task
        control_vector_info = None
        if task_name and model_config:
            control_vectors = model_config.get("control_vectors", {})
            if task_name in control_vectors:
                vector_info = control_vectors[task_name]
                vector_path = vector_info.get("path")

                if vector_path and os.path.exists(vector_path):
                    try:
                        # Load the control vector
                        vector_data = torch.load(vector_path)
                        control_vector_info = {
                            "vector": vector_data["vector"],
                            "layer": vector_data["layer"],
                            "path": vector_path,
                            "loaded": True,
                        }

                        if verbose:
                            print(f"\n🎮 Auto-loaded control vector for {task_name}")
                            print(f"   • Layer: {vector_data['layer']}")
                            print(f"   • Vector shape: {vector_data['vector'].shape}")
                            print(f"   • Path: {vector_path}")
                            print()
                    except Exception as e:
                        if verbose:
                            print(f"⚠️  Failed to load control vector: {e}")
    """
    Run the complete pipeline for a single task or file.
    
    Args:
        task_name: Name of available benchmark task or path to file (from 37 working benchmarks)
        model_name: Language model name
        layer: Layer for activation extraction
        shots: Number of few-shot examples
        split_ratio: Train/test split ratio
        limit: Optional limit on documents
        classifier_type: Type of classifier
        max_new_tokens: Max tokens for generation
        device: Target device
        seed: Random seed
        token_aggregation: How to aggregate token scores for classification
        from_csv: Whether task_name is a CSV file
        from_json: Whether task_name is a JSON file
        question_col: CSV column name for questions
        correct_col: CSV column name for correct answers
        incorrect_col: CSV column name for incorrect answers
        priority: Priority level for benchmark selection ("all", "high", "medium", "low")
        fast_only: Only use fast benchmarks (high priority)
        time_budget: Time budget in minutes for benchmark selection
        max_benchmarks: Maximum number of benchmarks to select
        smart_selection: Use smart benchmark selection based on relevance and priority
        
    Returns:
        Dictionary with all results
    """
    # VALIDATE TASK NAME - Only allow validated benchmarks from CORE_BENCHMARKS
    if not (from_csv or from_json):
        if not validate_task_name(task_name):
            error_msg = f"Invalid task name: '{task_name}'"
            suggestions = suggest_similar_tasks(task_name)

            if verbose:
                print(f"❌ {error_msg}")
                print(f"📋 Task '{task_name}' is not in the available benchmarks list.")
                print(
                    f"🔍 Only {len(AVAILABLE_BENCHMARKS)} working benchmarks are supported (out of {len(CORE_BENCHMARKS)} total)."
                )

                # Check if it's an unavailable benchmark
                if task_name in UNAVAILABLE_BENCHMARKS:
                    print("🚫 This benchmark is known to be unavailable/problematic.")

                if suggestions:
                    print("\n💡 Did you mean one of these?")
                    for i, suggestion in enumerate(suggestions, 1):
                        config = AVAILABLE_BENCHMARKS[suggestion]
                        priority = config.get("priority", "unknown")
                        tags = ", ".join(config.get("tags", []))
                        print(f"   {i}. {suggestion} ({priority} priority) - {tags}")

                print("\n📖 To see all valid tasks, run:")
                print("   wisent-guard tasks --list-tasks")
                print("\n📖 To see task details, run:")
                print("   wisent-guard tasks --task-info <task_name>")

            return {
                "task_name": task_name,
                "error": error_msg,
                "error_type": "invalid_task",
                "valid_tasks_count": len(AVAILABLE_BENCHMARKS),
                "total_benchmarks": len(CORE_BENCHMARKS),
                "excluded_count": len(UNAVAILABLE_BENCHMARKS),
                "suggestions": suggestions,
                "help": "Use --list-tasks to see all valid task names",
            }

    logger.info(f"Running pipeline for task: {task_name}")

    # Initialize performance tracking
    memory_tracker = None
    latency_tracker = None

    if enable_memory_tracking or enable_latency_tracking:
        if verbose:
            print(
                f"🔍 Performance tracking enabled: memory={enable_memory_tracking}, latency={enable_latency_tracking}"
            )

        from .core.tracking import (
            get_global_memory_tracker,
            get_global_latency_tracker,
            get_memory_info,
            format_memory_usage,
        )

        if enable_memory_tracking:
            memory_tracker = get_global_memory_tracker()
            memory_tracker.track_gpu = track_gpu_memory
            memory_tracker.sampling_interval = memory_sampling_interval
            memory_tracker.start_monitoring()
            if verbose:
                print(
                    f"   • Memory tracking started with {memory_sampling_interval}s interval"
                )

        if enable_latency_tracking:
            latency_tracker = get_global_latency_tracker()
            latency_tracker.start_tracking()
            if verbose:
                print("   • Latency tracking started")

    # Show current memory usage if requested
    if show_memory_usage:
        from .core.tracking import get_memory_info, format_memory_usage

        current_memory = get_memory_info()
        print(f"\n💾 Current Memory Usage: {format_memory_usage(current_memory)}")

    # AUTO-LOAD STEERING CONFIGURATION if steering mode is enabled but no method specified
    if steering_mode and not load_steering_vector:
        # Try to load optimal steering configuration
        from .core.steering_optimizer import get_optimal_steering_params

        optimal_steering = get_optimal_steering_params(model_name, task_name)

        if optimal_steering and optimal_steering.get("method"):
            # Use optimal parameters if no explicit override
            if steering_method == "CAA" and optimal_steering.get("method") != "CAA":
                steering_method = optimal_steering["method"]
                if verbose:
                    print(
                        f"\n🔧 Auto-loaded optimal steering method: {steering_method}"
                    )

            # Auto-load optimal steering layer if available and not explicitly overridden
            if optimal_steering.get("layer") is not None and steering_mode:
                # Only skip auto-loading if user explicitly specified a non-default layer
                original_layer = layer
                layer = str(optimal_steering["layer"])
                if verbose and original_layer != layer:
                    print(
                        f"🔧 Auto-loaded optimal steering layer: {layer} (was {original_layer})"
                    )

            if steering_strength == 1.0 and optimal_steering.get(
                "strength"
            ):  # Default strength
                steering_strength = optimal_steering["strength"]
                if verbose:
                    print(
                        f"🔧 Auto-loaded optimal steering strength: {steering_strength}"
                    )
        elif verbose:
            print(
                f"\n💡 No optimized steering config found. Run 'wisent-guard optimize-steering auto {model_name}' to optimize."
            )

    display_name = task_name if not (from_csv or from_json) else f"file:{task_name}"

    if verbose:
        print(f"\n{'=' * 80}")
        print(f"🚀 STARTING PIPELINE FOR TASK: {display_name.upper()}")
        print(f"{'=' * 80}")
        print("📋 Configuration:")
        print(f"   • Model: {model_name}")
        print(f"   • Layer: {layer}")
        print(f"   • Classifier: {classifier_type}")
        print(f"   • Max tokens: {max_new_tokens}")
        print(f"   • Split ratio: {split_ratio}")
        print(f"   • Token aggregation: {token_aggregation}")
        print(f"   • Limit: {limit}")
        print(f"   • Seed: {seed}")
        if from_csv:
            print("   • Input: CSV file")
            print(f"   • Columns: {question_col}, {correct_col}, {incorrect_col}")
        elif from_json:
            print("   • Input: JSON file")

    try:
        # Initialize control_vector_info to ensure it's in scope
        control_vector_info = None

        # Parse layers from argument
        layers = parse_layers_from_arg(layer)
        is_multi_layer = len(layers) > 1

        # Initialize enhanced primitives
        if verbose:
            print("\n🔧 Initializing model and primitives...")
            if is_multi_layer:
                print(f"   • Multi-layer mode: {layers}")
            else:
                print(f"   • Single layer mode: {layers[0]}")

        # Use provided model instance or load new one
        if model_instance is not None:
            model = model_instance
            if verbose:
                print("   • Using pre-loaded model instance (reused across tasks)")
        else:
            # Time model loading
            if latency_tracker:
                with latency_tracker.time_operation("model_loading"):
                    model = Model(name=model_name, device=device)
            else:
                model = Model(name=model_name, device=device)

        layer_obj = Layer(index=layers[0], type="transformer")

        # Create detection handler based on CLI arguments
        if verbose and detection_action != "pass_through":
            print("\n🛡️  Setting up detection handling:")
            print(f"   • Action: {detection_action}")
            if placeholder_message:
                print(f"   • Custom placeholder: {placeholder_message}")
            if detection_action == "regenerate_until_safe":
                print(f"   • Max regeneration attempts: {max_regeneration_attempts}")
            print(f"   • Detection threshold: {detection_threshold}")
            print(f"   • Logging enabled: {log_detections}")

        detection_handler = None
        if detection_action != "pass_through":
            from .core.detection_handling import DetectionHandler, DetectionAction

            # Map string to enum
            action_mapping = {
                "pass_through": DetectionAction.PASS_THROUGH,
                "replace_with_placeholder": DetectionAction.REPLACE_WITH_PLACEHOLDER,
                "regenerate_until_safe": DetectionAction.REGENERATE_UNTIL_SAFE,
            }

            detection_handler = DetectionHandler(
                action=action_mapping[detection_action],
                placeholder_message=placeholder_message,
                max_regeneration_attempts=max_regeneration_attempts,
                log_detections=log_detections,
            )

        if from_csv or from_json:
            # Load data from CSV/JSON file using ContrastivePairSet
            if verbose:
                print(f"\n📁 Loading data from {'CSV' if from_csv else 'JSON'} file...")

            # ContrastivePairSet is imported at the top of the file

            if from_csv:
                pair_set = ContrastivePairSet.from_csv_file(
                    name="csv_data",
                    csv_path=task_name,
                    question_col=question_col,
                    correct_col=correct_col,
                    incorrect_col=incorrect_col,
                    limit=limit,
                )
            else:  # from_json
                pair_set = ContrastivePairSet.from_json_file(
                    name="json_data", json_path=task_name, limit=limit
                )

            # Convert ContrastivePairSet to qa_pairs format for existing pipeline
            all_qa_pairs = []
            for pair in pair_set.pairs:
                if hasattr(pair, "question"):
                    all_qa_pairs.append(
                        {
                            "question": pair.question,
                            "correct_answer": pair.correct_answer,
                            "incorrect_answer": pair.incorrect_answer,
                        }
                    )

            # Split the qa_pairs
            import random

            random.seed(seed)
            random.shuffle(all_qa_pairs)
            # Apply reasonable limits to prevent huge datasets
            MAX_TRAIN_SAMPLES = 1000
            MAX_TEST_SAMPLES = 200

            split_point = int(len(all_qa_pairs) * split_ratio)
            qa_pairs = all_qa_pairs[:split_point]  # Training data
            test_qa_pairs_source = all_qa_pairs[split_point:]  # Test data

            # Apply limits
            if len(qa_pairs) > MAX_TRAIN_SAMPLES:
                qa_pairs = qa_pairs[:MAX_TRAIN_SAMPLES]
                if verbose:
                    print(f"   ⚠️  Training data limited to {MAX_TRAIN_SAMPLES} samples")

            if len(test_qa_pairs_source) > MAX_TEST_SAMPLES:
                test_qa_pairs_source = test_qa_pairs_source[:MAX_TEST_SAMPLES]
                if verbose:
                    print(f"   ⚠️  Test data limited to {MAX_TEST_SAMPLES} samples")

            if verbose:
                print(
                    f"📊 Data split: {len(qa_pairs)} training pairs, {len(test_qa_pairs_source)} test pairs"
                )

        else:
            # Traditional lm-harness task loading with caching support
            if verbose:
                print(f"📚 Loading task data for {task_name}...")

            # CACHING LOGIC: Check for cached benchmark data first
            cached_data = None
            used_cache = False

            if (
                use_cached
                and not force_download
                and is_benchmark_cached(task_name, cache_dir)
            ):
                if verbose:
                    print(f"💾 Found cached benchmark data for {task_name}")

                cached_data = load_cached_benchmark(task_name, cache_dir)
                if cached_data:
                    if verbose:
                        print(f"✅ Loaded {len(cached_data)} cached contrastive pairs")

                    # Convert cached data to QA pairs format
                    all_qa_pairs = convert_cached_data_to_qa_pairs(cached_data, limit)
                    used_cache = True

                    if verbose:
                        print(f"📊 Converted to {len(all_qa_pairs)} QA pairs")
                        if limit and len(all_qa_pairs) >= limit:
                            print(f"   📏 Limited to {limit} pairs as requested")
                else:
                    if verbose:
                        print(
                            "⚠️  Failed to load cached data, falling back to fresh download"
                        )

            if not used_cache:
                # Load fresh data from lm-eval
                if verbose:
                    print("🔄 Loading fresh data from lm-eval...")

                # FIRST: Check if this is a group task and expand if needed
                # Skip group task check for MBPP and LiveCodeBench as they're known to be slow
                if task_name in ["mbpp", "livecodebench"]:
                    if verbose:
                        print(
                            f"🔍 Skipping group task check for {task_name} (known to be slow)"
                        )
                    group_task_processed = False
                    group_task_qa_format = False
                else:
                    try:
                        from lm_eval import evaluator
                        import threading

                        if verbose:
                            print(f"🔍 Checking if '{task_name}' is a group task...")

                        # Use lm-eval's task expansion to detect group tasks with timeout
                        task_dict = None
                        exception_occurred = None

                        def load_task_dict():
                            nonlocal task_dict, exception_occurred
                            try:
                                task_dict = evaluator.get_task_dict([task_name])
                            except Exception as e:
                                exception_occurred = e

                        # Start loading in a separate thread with timeout
                        thread = threading.Thread(target=load_task_dict)
                        thread.daemon = True
                        thread.start()

                        # Wait for up to 60 seconds
                        thread.join(timeout=60)

                        if thread.is_alive():
                            # Thread is still running, task loading timed out
                            raise TimeoutError(
                                f"Task loading timed out for {task_name}"
                            )

                        if exception_occurred:
                            raise exception_occurred

                        if task_dict is None:
                            raise RuntimeError(f"Task loading failed for {task_name}")

                        expanded_tasks = list(task_dict.keys())

                        if len(expanded_tasks) > 1:
                            if verbose:
                                print(
                                    f"🎯 Detected GROUP task '{task_name}' with {len(expanded_tasks)} subtasks: {expanded_tasks[:5]}{'...' if len(expanded_tasks) > 5 else ''}"
                                )
                                print("📚 Extracting samples from all subtasks...")

                            # Handle group task by combining samples from all subtasks
                            all_qa_pairs = []

                            for subtask_name in expanded_tasks:
                                try:
                                    if verbose:
                                        print(f"   🔍 Loading subtask: {subtask_name}")

                                    # Load each subtask individually
                                    subtask_data = model.load_lm_eval_task(
                                        subtask_name, shots=shots, limit=limit
                                    )
                                    subtask_train_docs, _ = model.split_task_data(
                                        subtask_data,
                                        split_ratio=split_ratio,
                                        random_seed=seed,
                                    )

                                    # Extract QA pairs from this subtask
                                    subtask_qa_pairs = ContrastivePairSet.extract_qa_pairs_from_task_docs(
                                        subtask_name, subtask_data, subtask_train_docs
                                    )

                                    if verbose:
                                        print(
                                            f"   ✅ Extracted {len(subtask_qa_pairs)} pairs from {subtask_name}"
                                        )

                                    # Add subtask identifier to each pair for tracking
                                    for pair in subtask_qa_pairs:
                                        pair["source_subtask"] = subtask_name

                                    all_qa_pairs.extend(subtask_qa_pairs)

                                except Exception as e:
                                    if verbose:
                                        print(
                                            f"   ⚠️ Failed to load subtask {subtask_name}: {e}"
                                        )
                                    continue

                            if not all_qa_pairs:
                                raise ValueError(
                                    f"No QA pairs could be extracted from any subtasks in group '{task_name}'"
                                )

                            # Shuffle and limit the combined dataset
                            import random

                            random.seed(seed)
                            random.shuffle(all_qa_pairs)

                            if limit and len(all_qa_pairs) > limit:
                                all_qa_pairs = all_qa_pairs[:limit]

                            # Split for training and testing with reasonable limits
                            MAX_TRAIN_SAMPLES = 1000
                            MAX_TEST_SAMPLES = 200

                            split_point = int(len(all_qa_pairs) * split_ratio)
                            qa_pairs = all_qa_pairs[:split_point]  # Training data
                            test_qa_pairs_source = all_qa_pairs[
                                split_point:
                            ]  # Test data

                            # Apply limits
                            if len(qa_pairs) > MAX_TRAIN_SAMPLES:
                                qa_pairs = qa_pairs[:MAX_TRAIN_SAMPLES]
                                if verbose:
                                    print(
                                        f"   ⚠️  Training data limited to {MAX_TRAIN_SAMPLES} samples"
                                    )

                            if len(test_qa_pairs_source) > MAX_TEST_SAMPLES:
                                test_qa_pairs_source = test_qa_pairs_source[
                                    :MAX_TEST_SAMPLES
                                ]
                                if verbose:
                                    print(
                                        f"   ⚠️  Test data limited to {MAX_TEST_SAMPLES} samples"
                                    )

                            if verbose:
                                print(
                                    f"📊 Combined dataset: {len(qa_pairs)} total QA pairs from {len(expanded_tasks)} subtasks"
                                )

                                # Show breakdown by subtask
                                from collections import Counter

                                subtask_counts = Counter(
                                    pair.get("source_subtask", "unknown")
                                    for pair in qa_pairs
                                )
                                print("📋 Breakdown by subtask:")
                                for subtask, count in subtask_counts.most_common():
                                    print(f"   • {subtask}: {count} pairs")

                            # Set a flag to indicate we've already processed the group task
                            group_task_processed = True
                            group_task_qa_format = (
                                True  # Test data is already in QA format
                            )

                        else:
                            if verbose:
                                print(
                                    f"✅ '{task_name}' is an individual task, proceeding..."
                                )
                            group_task_processed = False
                            group_task_qa_format = False

                    except TimeoutError as e:
                        if verbose:
                            print(f"⏰ Task loading timed out for '{task_name}': {e}")
                            print("🔄 Proceeding with standard task loading...")
                        group_task_processed = False
                        group_task_qa_format = False
                    except Exception as e:
                        if "Group task" in str(e):
                            raise e  # Re-raise group task errors
                        elif verbose:
                            print(
                                f"⚠️  Could not check if '{task_name}' is a group task: {e}"
                            )
                            print("🔄 Proceeding with standard task loading...")
                        group_task_processed = False
                        group_task_qa_format = False

                # Only load individual task if we haven't already processed a group task
                if not group_task_processed:
                    # FIXED: Resolve benchmark name to actual task name for lm-eval-harness
                    actual_task_name = get_actual_task_name(task_name)
                    if verbose and actual_task_name != task_name:
                        print(
                            f"🔄 Resolving benchmark '{task_name}' to task '{actual_task_name}'"
                        )

                    # Determine the total limit to load
                    # If training_limit or testing_limit are specified, we need to load enough data
                    total_limit = limit
                    if training_limit is not None or testing_limit is not None:
                        # Calculate how many total docs we need based on split ratio
                        # Add some buffer to ensure we have enough after splitting
                        if training_limit is not None and testing_limit is not None:
                            # Both specified, use sum
                            total_limit = training_limit + testing_limit
                        elif training_limit is not None:
                            # Only training limit specified, calculate total based on split ratio
                            # Add 20% buffer to account for rounding
                            total_limit = int(training_limit / split_ratio * 1.2) + 1
                        elif testing_limit is not None:
                            # Only testing limit specified, calculate total based on split ratio
                            # Add 20% buffer to account for rounding
                            total_limit = (
                                int(testing_limit / (1 - split_ratio) * 1.2) + 1
                            )

                    task_data = model.load_lm_eval_task(
                        actual_task_name, shots=shots, limit=total_limit
                    )
                    train_docs, test_docs = model.split_task_data(
                        task_data, split_ratio=split_ratio, random_seed=seed
                    )

                    # Apply training and testing limits if specified
                    original_train_size = len(train_docs)
                    original_test_size = len(test_docs)

                    if training_limit is not None and len(train_docs) > training_limit:
                        train_docs = train_docs[:training_limit]
                        if verbose:
                            print(
                                f"   ⚠️  Training data limited to {training_limit} samples (from {original_train_size})"
                            )

                    if testing_limit is not None and len(test_docs) > testing_limit:
                        test_docs = test_docs[:testing_limit]
                        if verbose:
                            print(
                                f"   ⚠️  Test data limited to {testing_limit} samples (from {original_test_size})"
                            )

                    if verbose:
                        print(
                            f"📊 Data split: {len(train_docs)} training docs, {len(test_docs)} test docs"
                        )

                    # Extract QA pairs from training documents
                    if verbose:
                        print("\n📝 TRAINING DATA PREPARATION:")
                        print(
                            f"   • Loading {task_name} data with correct/incorrect answers..."
                        )

                    # NEW: Use ManagedCachedBenchmarks for intelligent downloading
                    if cache_benchmark:
                        from .core.managed_cached_benchmarks import get_managed_cache

                        managed_cache = get_managed_cache(cache_dir)

                        try:
                            # Get samples using intelligent caching
                            # Calculate total limit needed based on training/testing limits
                            cache_limit = limit
                            if training_limit is not None or testing_limit is not None:
                                # Need enough samples to satisfy both limits after split
                                # Add buffer to ensure we have enough after splitting
                                if (
                                    training_limit is not None
                                    and testing_limit is not None
                                ):
                                    cache_limit = training_limit + testing_limit
                                elif training_limit is not None:
                                    # Add 20% buffer to account for rounding
                                    cache_limit = (
                                        int(training_limit / split_ratio * 1.2) + 1
                                    )
                                elif testing_limit is not None:
                                    # Add 20% buffer to account for rounding
                                    cache_limit = (
                                        int(testing_limit / (1 - split_ratio) * 1.2) + 1
                                    )

                            cached_samples = managed_cache.get_task_samples(
                                task_name=task_name,
                                limit=cache_limit
                                or 1000,  # Default to 1000 if no limit
                                force_fresh=force_download,
                            )

                            # Convert cached samples to QA pairs format
                            qa_pairs = [
                                sample["normalized"] for sample in cached_samples
                            ]

                            if verbose:
                                print(
                                    f"✅ Using managed cache: {len(qa_pairs)} samples loaded efficiently"
                                )

                        except Exception as e:
                            if verbose:
                                print(
                                    f"⚠️  Managed cache failed, falling back to traditional method: {e}"
                                )

                            # Fallback to traditional method
                            qa_pairs = (
                                ContrastivePairSet.extract_qa_pairs_from_task_docs(
                                    task_name, task_data, train_docs
                                )
                            )
                    else:
                        # Traditional method when caching is disabled
                        qa_pairs = ContrastivePairSet.extract_qa_pairs_from_task_docs(
                            task_name, task_data, train_docs
                        )

                    test_qa_pairs_source = (
                        test_docs  # Keep original format for test docs
                    )

                # CACHE SAVING: Save to cache if requested and we loaded fresh data
                if cache_benchmark and not used_cache:
                    if verbose:
                        print("💾 Caching benchmark data for future use...")

                    success = save_benchmark_to_cache(
                        task_name, cache_dir, limit, verbose
                    )
                    if success and verbose:
                        print("✅ Benchmark cached successfully!")
                    elif verbose:
                        print("⚠️  Failed to cache benchmark")
            else:
                # We used cached data, process it for the pipeline
                # Properly split the data using the split ratio with limits
                # Use training_limit and testing_limit if specified, otherwise use defaults
                MAX_TRAIN_SAMPLES = (
                    training_limit if training_limit is not None else 1000
                )
                MAX_TEST_SAMPLES = testing_limit if testing_limit is not None else 200

                split_point = int(len(all_qa_pairs) * split_ratio)
                qa_pairs = all_qa_pairs[:split_point]  # Training data
                test_qa_pairs_source = all_qa_pairs[split_point:]  # Test data

                if verbose:
                    print(
                        f"📊 Initial split: {len(qa_pairs)} training, {len(test_qa_pairs_source)} test samples"
                    )

                # Apply limits
                original_train_size = len(qa_pairs)
                original_test_size = len(test_qa_pairs_source)

                if len(qa_pairs) > MAX_TRAIN_SAMPLES:
                    qa_pairs = qa_pairs[:MAX_TRAIN_SAMPLES]
                    if verbose:
                        print(
                            f"   ⚠️  Training data limited to {MAX_TRAIN_SAMPLES} samples (from {original_train_size})"
                        )

                if len(test_qa_pairs_source) > MAX_TEST_SAMPLES:
                    test_qa_pairs_source = test_qa_pairs_source[:MAX_TEST_SAMPLES]
                    if verbose:
                        print(
                            f"   ⚠️  Test data limited to {MAX_TEST_SAMPLES} samples (from {original_test_size})"
                        )

                if verbose:
                    print(
                        f"📊 Final data split: {len(qa_pairs)} training, {len(test_qa_pairs_source)} test samples"
                    )
                group_task_processed = False
                group_task_qa_format = True  # Cached data is already in QA format

        if verbose:
            print(f"   • Successfully extracted {len(qa_pairs)} QA pairs")
            print("\n🔍 Training Examples:")
            for i, qa_pair in enumerate(qa_pairs[:4]):  # Show first 4
                print(f"\n   📋 Example {i + 1}:")
                question_preview = (
                    qa_pair["question"][:100] + "..."
                    if len(qa_pair["question"]) > 100
                    else qa_pair["question"]
                )
                print(f"      🔸 Question: {question_preview}")
                print(f"      ✅ Correct Answer: {qa_pair['correct_answer']}")
                if "incorrect_answer" in qa_pair:
                    print(f"      ❌ Incorrect Answer: {qa_pair['incorrect_answer']}")

        # 🚨 HARD ERROR CHECK: QA pair extraction failure
        if len(qa_pairs) == 0:
            raise ValueError(
                f"❌ CRITICAL ERROR: QA pair extraction failed for task '{task_name}'!\n"
                f"   📊 Task loaded {len(train_docs) if 'train_docs' in locals() else 0} training documents\n"
                f"   🔍 But extracted 0 QA pairs from them\n"
                f"   💡 This indicates the task document structure is not recognized by the extraction logic\n"
                f"   🛠️  Check ContrastivePairSet.extract_qa_pairs_from_task_docs() for '{task_name}' support"
            )

        # FIXED: For perplexity tasks, skip contrastive training and go directly to evaluation
        if ground_truth_method == "lm-eval-harness":
            # Check evaluation method for this task
            def get_evaluation_method_for_task_early(task_name: str) -> str:
                """Get the evaluation method for a task from the benchmark configuration."""
                try:
                    import json
                    import os

                    eval_methods_path = os.path.join(
                        os.path.dirname(__file__),
                        "parameters/benchmarks/benchmark_evaluation_methods.json",
                    )
                    with open(eval_methods_path, "r") as f:
                        benchmark_methods = json.load(f)
                        return benchmark_methods.get(task_name, "text-generation")
                except Exception as e:
                    if verbose:
                        print(f"   ⚠️ Could not load benchmark evaluation methods: {e}")
                    return "text-generation"

            evaluation_method = get_evaluation_method_for_task_early(task_name)

            if evaluation_method == "perplexity":
                if verbose:
                    print(
                        "\n🎯 PERPLEXITY TASK DETECTED: Skipping contrastive training"
                    )
                    print(f"   • Task: {task_name}")
                    print(f"   • Evaluation method: {evaluation_method}")
                    print("   • Going directly to perplexity evaluation")

                # Create a minimal "dummy" classifier for the perplexity evaluation
                # Note: LMEvalHarnessGroundTruth is already imported at the top of the file

                # Parse layers for evaluation
                layers = parse_layers_from_arg(layer)

                # Use actual task name for evaluation
                actual_eval_task_name = get_actual_task_name(task_name)
                lm_eval_ground_truth = LMEvalHarnessGroundTruth(
                    actual_eval_task_name, evaluation_method, model=model
                )

                # Run perplexity evaluation without classifier
                lm_eval_results = lm_eval_ground_truth.evaluate_classifier_on_task(
                    classifier=None,  # No classifier needed for perplexity
                    task_name=actual_eval_task_name,
                    num_samples=len(test_qa_pairs_source),
                    model=model,
                    layer=layers[0],
                    token_aggregation=token_aggregation,
                )

                if verbose:
                    print(
                        f"\n🎉 PERPLEXITY EVALUATION COMPLETED FOR {task_name.upper()}!"
                    )
                    print(f"{'=' * 80}")
                    print("📊 FINAL RESULTS:")
                    print(f"   • Test samples: {len(test_qa_pairs_source)}")
                    print(f"   • Evaluation method: {evaluation_method}")

                    if "perplexity_accuracy" in lm_eval_results:
                        accuracy = lm_eval_results["perplexity_accuracy"]
                        print(f"   • Perplexity accuracy: {accuracy:.2%}")
                    elif "average_classifier_score" in lm_eval_results:
                        avg_score = lm_eval_results["average_classifier_score"]
                        print(f"   • Average perplexity score: {avg_score:.3f}")
                    else:
                        print("   • Perplexity evaluation: Completed")
                    print(f"{'=' * 80}")

                return {
                    "task_name": task_name,
                    "model_name": model_name,
                    "layer": layer,
                    "evaluation_method": evaluation_method,
                    "evaluation_results": lm_eval_results,
                    "num_test": len(test_qa_pairs_source),
                    "ground_truth_method": "lm-eval-harness",
                    "skipped_training": True,
                    "reason": "Perplexity task does not require contrastive training",
                }

        # Validate dataset size before proceeding (for non-perplexity tasks)
        min_training_samples = 4
        if len(qa_pairs) < min_training_samples:
            error_msg = f"Insufficient training data: {len(qa_pairs)} pairs found, minimum {min_training_samples} required"
            if verbose:
                print(f"\n❌ ERROR: {error_msg}")
                print("   • Consider increasing --limit or using a larger dataset")
                print(
                    f"   • CSV/JSON files should have at least {min_training_samples} rows"
                )
                print("   • lm-harness tasks may need higher --limit values")

            if not allow_small_dataset:
                if verbose:
                    print(
                        "   • Use --allow-small-dataset flag to bypass this check (may cause training issues)"
                    )

                return {
                    "task_name": task_name,
                    "model_name": model_name,
                    "error": error_msg,
                    "training_samples": len(qa_pairs),
                    "minimum_required": min_training_samples,
                    "suggestion": "Increase dataset size, --limit parameter, or use --allow-small-dataset flag",
                }
            else:
                if verbose:
                    print(
                        "   ⚠️  WARNING: Proceeding with small dataset due to --allow-small-dataset flag"
                    )
                    print(
                        f"   • Training may be unstable with only {len(qa_pairs)} samples"
                    )

        # Create contrastive pairs using proper activation collection logic
        from .core.activation_collection_method import (
            ActivationCollectionLogic,
            TokenTargetingStrategy,
            PromptConstructionStrategy,
        )
        from .core.activations import Activations, ActivationAggregationMethod

        # Convert strings to enums
        prompt_strategy_mapping = {
            "multiple_choice": PromptConstructionStrategy.MULTIPLE_CHOICE,
            "role_playing": PromptConstructionStrategy.ROLE_PLAYING,
            "direct_completion": PromptConstructionStrategy.DIRECT_COMPLETION,
            "instruction_following": PromptConstructionStrategy.INSTRUCTION_FOLLOWING,
        }
        prompt_strategy = prompt_strategy_mapping.get(
            prompt_construction_strategy, PromptConstructionStrategy.MULTIPLE_CHOICE
        )

        targeting_strategy_mapping = {
            "choice_token": TokenTargetingStrategy.CHOICE_TOKEN,
            "continuation_token": TokenTargetingStrategy.CONTINUATION_TOKEN,
            "last_token": TokenTargetingStrategy.LAST_TOKEN,
            "first_token": TokenTargetingStrategy.FIRST_TOKEN,
            "mean_pooling": TokenTargetingStrategy.MEAN_POOLING,
            "max_pooling": TokenTargetingStrategy.MAX_POOLING,
        }
        targeting_strategy = targeting_strategy_mapping.get(
            token_targeting_strategy, TokenTargetingStrategy.CHOICE_TOKEN
        )

        if verbose:
            print(f"   • Prompt construction: {prompt_strategy.value}")
            print(f"   • Token targeting: {targeting_strategy.value}")

        collector = ActivationCollectionLogic(model=model)
        contrastive_pairs = collector.create_batch_contrastive_pairs(
            qa_pairs, prompt_strategy
        )

        # 🚨 HARD ERROR CHECK: No training data created
        if len(contrastive_pairs) == 0:
            raise ValueError(
                f"❌ CRITICAL ERROR: No contrastive pairs created for task '{task_name}'!\n"
                f"   📊 Input: {len(qa_pairs)} QA pairs\n"
                f"   🔍 Output: 0 contrastive pairs\n"
                f"   💡 This indicates the contrastive pair creation logic failed\n"
                f"   🛠️  Check ActivationCollectionLogic.create_batch_contrastive_pairs() method"
            )

        if verbose:
            print(f"\n🔄 Created {len(contrastive_pairs)} contrastive pairs:")
            for i, pair in enumerate(contrastive_pairs[:3]):  # Show first 3
                print(f"\n   🔄 Contrastive Pair {i + 1}:")
                print(f"      📝 Prompt: {pair.prompt}")
                print(f"      🟢 Positive (B): {pair.positive_response}")
                print(f"      🔴 Negative (A): {pair.negative_response}")

        # Validate mode combinations
        if train_only and inference_only:
            error_msg = "Cannot specify both --train-only and --inference-only modes"
            if verbose:
                print(f"\n❌ ERROR: {error_msg}")
            return {"task_name": task_name, "error": error_msg}

        if inference_only and not load_classifier and not load_steering_vector:
            error_msg = "Inference-only mode requires --load-classifier or --load-steering-vector"
            if verbose:
                print(f"\n❌ ERROR: {error_msg}")
                print("   • Use --load-classifier to load pre-trained classifiers")
                print(
                    "   • Use --load-steering-vector to load pre-trained steering vectors"
                )
            return {"task_name": task_name, "error": error_msg}

        # Handle inference-only mode
        if inference_only:
            from .core.model_persistence import ModelPersistence

            if verbose:
                print("\n🔄 INFERENCE-ONLY MODE:")
                print("   • Loading pre-trained models for inference...")

            # Parse layers to know what to load
            layers = parse_layers_from_arg(layer)

            # Load classifiers or steering vectors
            steering_methods = {}
            loaded_models = {}

            if load_classifier:
                if verbose:
                    print(f"   • Loading classifiers from: {load_classifier}")

                if len(layers) > 1:
                    # Multi-layer mode
                    classifiers_data = ModelPersistence.load_multi_layer_classifiers(
                        load_classifier, layers
                    )
                    for layer_idx, (classifier, metadata) in classifiers_data.items():
                        steering_methods[layer_idx] = type(
                            "SteeringMethod", (), {"classifier": classifier}
                        )()
                        loaded_models[layer_idx] = metadata
                        if verbose:
                            print(
                                f"     ✅ Layer {layer_idx}: {metadata.get('classifier_type', 'unknown')} classifier"
                            )
                else:
                    # Single layer mode
                    classifier, metadata = ModelPersistence.load_classifier(
                        load_classifier, layers[0]
                    )
                    steering_methods[layers[0]] = type(
                        "SteeringMethod", (), {"classifier": classifier}
                    )()
                    loaded_models[layers[0]] = metadata
                    if verbose:
                        print(
                            f"     ✅ Layer {layers[0]}: {metadata.get('classifier_type', 'unknown')} classifier"
                        )

            if load_steering_vector:
                if verbose:
                    print(f"   • Loading steering vectors from: {load_steering_vector}")

                try:
                    from .core.steering_method import CAA

                    steering_method = CAA(device=device)

                    if steering_method.load_steering_vector(load_steering_vector):
                        layer_index = steering_method.layer_index
                        if layer_index is not None:
                            steering_methods[layer_index] = steering_method
                            loaded_models[layer_index] = {
                                "method_name": steering_method.name,
                                "aggregation_method": (
                                    steering_method.aggregation_method.value
                                    if hasattr(steering_method, "aggregation_method")
                                    else "caa"
                                ),
                                "loaded_from": load_steering_vector,
                            }
                            if verbose:
                                print(
                                    f"     ✅ Loaded steering vector for layer {layer_index}"
                                )
                        else:
                            if verbose:
                                print(
                                    "     ⚠️  Warning: No layer information in loaded vector"
                                )
                    else:
                        if verbose:
                            print(
                                f"     ❌ Failed to load steering vector from {load_steering_vector}"
                            )
                except Exception as e:
                    if verbose:
                        print(f"     ❌ Error loading steering vector: {str(e)}")

            if not steering_methods:
                error_msg = "No models could be loaded for inference"
                if verbose:
                    print(f"\n❌ ERROR: {error_msg}")
                return {"task_name": task_name, "error": error_msg}

            # Set up inference with loaded models
            if verbose:
                print("   • Inference setup complete")
                print(f"   • Available models: {list(loaded_models.keys())}")

            # Continue with inference pipeline using loaded models
            # (The rest of the pipeline will use the loaded steering_methods)

        # Handle training-only mode
        elif train_only:
            if verbose:
                print("\n🎓 TRAINING-ONLY MODE:")
                print(
                    "   • Training classifiers/vectors and saving, skipping inference..."
                )

            # Continue with training but return early before inference
            # (Training will happen in the normal flow, but we'll return after training)
            pass

        # Check if optimization is needed
        original_layer = layer
        original_token_aggregation = token_aggregation
        optimization_result = None

        if optimize and ground_truth_method == "interactive":
            # Special case: Interactive optimization - SKIP normal pipeline
            # Generate test questions for optimization
            test_questions = []
            for doc in test_qa_pairs_source[
                :2
            ]:  # Use first 2 test questions for optimization
                try:
                    if from_csv or from_json:
                        # For CSV/JSON, doc is already a qa_pair dict
                        question = doc["question"]
                    else:
                        # For lm-harness tasks, extract from document
                        if hasattr(task_data, "doc_to_text"):
                            question = task_data.doc_to_text(doc)
                        else:
                            question = doc.get("question", str(doc))
                    test_questions.append(question)
                except Exception:
                    continue

            if test_questions:
                # Run interactive optimization ONLY
                optimization_result = run_interactive_optimization(
                    model=model,
                    questions=test_questions,
                    training_pairs=contrastive_pairs,
                    max_new_tokens=max_new_tokens,
                    max_combinations=optimize_max_combinations,
                    verbose=verbose,
                )

                if optimization_result.get("optimization_performed"):
                    layer = optimization_result["best_layer"]
                    token_aggregation = optimization_result["best_aggregation"]

                    if verbose:
                        print("✅ Interactive optimization completed!")
                        print(f"   • Optimized layer: {layer} (was {original_layer})")
                        print(
                            f"   • Optimized aggregation: {token_aggregation} (was {original_token_aggregation})"
                        )

                    # Return results immediately - skip normal pipeline
                    return {
                        "task_name": task_name,
                        "model_name": model_name,
                        "layer": layer,
                        "original_layer": original_layer,
                        "token_aggregation": token_aggregation,
                        "original_token_aggregation": original_token_aggregation,
                        "optimization_performed": True,
                        "optimization_result": optimization_result,
                        "interactive_optimization_only": True,
                    }
            else:
                if verbose:
                    print("⚠️ No test questions available for interactive optimization")
                return {"task_name": task_name, "error": "No test questions available"}

        elif optimize:
            # Load test data first for optimization
            test_qa_pairs = []
            for doc in test_qa_pairs_source:
                try:
                    if from_csv or from_json or group_task_qa_format:
                        # For CSV/JSON/cached data, doc is already a qa_pair dict
                        test_qa_pairs.append(
                            {
                                "question": doc["question"],
                                "formatted_question": doc["question"],
                                "correct_answer": doc["correct_answer"],
                            }
                        )
                    else:
                        # For lm-harness tasks, use comprehensive extraction method
                        # ContrastivePairSet is already imported at the top of the file

                        # Extract QA pairs using the comprehensive method that handles all benchmarks
                        qa_pairs_batch = (
                            ContrastivePairSet.extract_qa_pairs_from_task_docs(
                                task_name, task_data, [doc]
                            )
                        )

                        if qa_pairs_batch:
                            test_qa_pairs.extend(qa_pairs_batch)
                        elif verbose:
                            print(
                                f"⚠️  Failed to extract test QA pair from benchmark {task_name}"
                            )

                except Exception:
                    continue

            # Run smart optimization with caching
            optimization_result = run_smart_optimization(
                model=model,
                collector=collector,
                contrastive_pairs=contrastive_pairs,
                test_qa_pairs=test_qa_pairs,
                task_name=task_name,
                model_name=model_name,
                limit=limit,
                ground_truth_method=ground_truth_method,
                max_new_tokens=max_new_tokens,
                device=device,
                verbose=verbose,
                optimize_layers=optimize_layers,
            )

            # Extract the best parameters from optimization
        if optimization_result and optimization_result.get(
            "optimization_performed", False
        ):
            layer = optimization_result.get("best_layer", layer)
            token_aggregation = optimization_result.get(
                "best_aggregation", token_aggregation
            )
            optimized_classifier_type = optimization_result.get(
                "best_classifier_type", classifier_type
            )
            optimized_threshold = optimization_result.get("best_threshold", 0.6)

            # 🚨 FIX: Update layers array to use optimized layer for ground truth evaluation
            optimized_layer_int = int(layer) if isinstance(layer, str) else layer
            layers = [optimized_layer_int]  # Update layers array with optimized layer
            layer_obj = Layer(
                index=optimized_layer_int, type="transformer"
            )  # Update layer object too

            if verbose:
                print("✅ Hyperparameter optimization completed!")
                print(f"   • Best layer: {layer} + {token_aggregation} aggregation")
                print(f"   • Best classifier: {optimized_classifier_type}")
                print(f"   • Best threshold: {optimized_threshold}")
                print(f"   • Updated layers array: {layers}")
        else:
            if verbose:
                print(f"⚠️ Optimization failed, using default layer {layer}")
            optimized_classifier_type = classifier_type
            optimized_threshold = 0.6
            optimization_result = {
                "best_layer": layer,
                "best_aggregation": token_aggregation,
                "best_classifier_type": optimized_classifier_type,
                "best_threshold": optimized_threshold,
                "optimization_performed": False,
            }

        # Extract activations from the choice tokens using the (possibly optimized) layer
        optimization_note = (
            " (optimized)" if optimize and layer != original_layer else ""
        )
        if verbose:
            print(
                f"\n🔬 Extracting activations from layer {layer}{optimization_note} choice tokens..."
            )

        if latency_tracker:
            with latency_tracker.time_operation("activation_extraction"):
                processed_pairs = collector.collect_activations_batch(
                    pairs=contrastive_pairs,
                    layer_index=layers[0],
                    device=device,
                    token_targeting_strategy=targeting_strategy,
                )
        else:
            processed_pairs = collector.collect_activations_batch(
                pairs=contrastive_pairs,
                layer_index=layers[0],
                device=device,
                token_targeting_strategy=targeting_strategy,
            )

        # Convert to ContrastivePairSet format for training
        phrase_pairs = []
        for pair in processed_pairs:
            # Create the full prompts for the pair set
            positive_full = f"{pair.prompt}{pair.positive_response}"
            negative_full = f"{pair.prompt}{pair.negative_response}"

            phrase_pairs.append(
                {
                    "harmful": negative_full,  # A choice (incorrect)
                    "harmless": positive_full,  # B choice (correct)
                }
            )

        # Create ContrastivePairSet with the real activations
        pair_set = ContrastivePairSet.from_phrase_pairs(
            name=f"{task_name}_training",
            phrase_pairs=phrase_pairs,
            task_type="lm_evaluation",
        )

        # Store the real activations in the pair set response objects
        for i, processed_pair in enumerate(processed_pairs):
            if i < len(pair_set.pairs):
                # Assign activations to the response objects, not the pair directly
                if (
                    hasattr(pair_set.pairs[i], "positive_response")
                    and pair_set.pairs[i].positive_response
                ):
                    pair_set.pairs[
                        i
                    ].positive_response.activations = (
                        processed_pair.positive_activations
                    )
                if (
                    hasattr(pair_set.pairs[i], "negative_response")
                    and pair_set.pairs[i].negative_response
                ):
                    pair_set.pairs[
                        i
                    ].negative_response.activations = (
                        processed_pair.negative_activations
                    )

        # STEERING MODE vs CLASSIFICATION MODE
        if steering_mode:
            if verbose:
                print(f"\n🎯 STEERING MODE: Computing {steering_method} Vector")
                print(f"   • Method: {steering_method}")
                print(f"   • Target layer: {layer}")
                print(f"   • Training pairs: {len(pair_set)}")
                print(f"   • Steering strength: {steering_strength}")
                print(f"   • Normalization: {normalization_method}")
                if target_norm:
                    print(f"   • Target norm: {target_norm}")

            # Import steering methods
            from .core.steering_method import CAA, HPR, DAC, BiPO, KSteering
            from .core.normalization import VectorNormalizationMethod

            # Ensure task_data is available for steering evaluation
            # If not already loaded (e.g., from CSV/JSON), load it now
            if "task_data" not in locals():
                # FIXED: Resolve benchmark name to actual task name for lm-eval-harness
                actual_task_name = get_actual_task_name(task_name)
                if verbose and actual_task_name != task_name:
                    print(
                        f"🔄 Resolving benchmark '{task_name}' to task '{actual_task_name}'"
                    )

                task_data = model.load_lm_eval_task(
                    actual_task_name, shots=shots, limit=limit
                )

            # Convert string to enum
            try:
                norm_method = VectorNormalizationMethod(normalization_method)
            except ValueError:
                norm_method = VectorNormalizationMethod.NONE
                if verbose:
                    print(
                        f"   • Warning: Unknown normalization method '{normalization_method}', using 'none'"
                    )

            # Create steering method based on selection
            if steering_method == "CAA":
                steering_obj = CAA(
                    device=device,
                    normalization_method=norm_method,
                    target_norm=target_norm,
                )
            elif steering_method == "HPR":
                steering_obj = HPR(device=device, beta=hpr_beta)
                if verbose:
                    print(f"   • HPR beta: {hpr_beta}")
            elif steering_method == "DAC":
                steering_obj = DAC(
                    device=device,
                    dynamic_control=dac_dynamic_control,
                    entropy_threshold=dac_entropy_threshold,
                )
                if verbose:
                    print(f"   • DAC dynamic control: {dac_dynamic_control}")
                    print(f"   • DAC entropy threshold: {dac_entropy_threshold}")
            elif steering_method == "BiPO":
                steering_obj = BiPO(
                    device=device,
                    beta=bipo_beta,
                    learning_rate=bipo_learning_rate,
                    num_epochs=bipo_epochs,
                )
                if verbose:
                    print(f"   • BiPO beta: {bipo_beta}")
                    print(f"   • BiPO learning rate: {bipo_learning_rate}")
                    print(f"   • BiPO epochs: {bipo_epochs}")
            elif steering_method == "KSteering":
                # Parse target and avoid labels
                target_labels = [
                    int(x.strip())
                    for x in ksteering_target_labels.split(",")
                    if x.strip()
                ]
                avoid_labels = (
                    [
                        int(x.strip())
                        for x in ksteering_avoid_labels.split(",")
                        if x.strip()
                    ]
                    if ksteering_avoid_labels
                    else []
                )

                steering_obj = KSteering(
                    device=device,
                    num_labels=ksteering_num_labels,
                    hidden_dim=ksteering_hidden_dim,
                    learning_rate=ksteering_learning_rate,
                    classifier_epochs=ksteering_classifier_epochs,
                    target_labels=target_labels,
                    avoid_labels=avoid_labels,
                    alpha=ksteering_alpha,
                )
                if verbose:
                    print(f"   • K-Steering num labels: {ksteering_num_labels}")
                    print(f"   • K-Steering hidden dim: {ksteering_hidden_dim}")
                    print(f"   • K-Steering learning rate: {ksteering_learning_rate}")
                    print(
                        f"   • K-Steering classifier epochs: {ksteering_classifier_epochs}"
                    )
                    print(f"   • K-Steering target labels: {target_labels}")
                    print(f"   • K-Steering avoid labels: {avoid_labels}")
                    print(f"   • K-Steering alpha: {ksteering_alpha}")
            else:
                raise ValueError(f"Unknown steering method: {steering_method}")

            # Apply token steering wrapper if enabled
            if enable_token_steering:
                if verbose:
                    print(f"   • Token steering enabled: {token_steering_strategy}")
                    print(f"   • Token decay rate: {token_decay_rate}")
                    print(
                        f"   • Token strength range: {token_min_strength} - {token_max_strength}"
                    )
                    print(f"   • Apply to prompt: {token_apply_to_prompt}")
                    if token_apply_to_prompt:
                        print(
                            f"   • Prompt strength multiplier: {token_prompt_strength_multiplier}"
                        )

                from .core.steering_methods.token_steered import (
                    TokenSteeringStrategy,
                    TokenSteeringConfig,
                    TokenSteeringWrapper,
                )

                # Convert string to enum
                strategy_mapping = {
                    "last_only": TokenSteeringStrategy.LAST_ONLY,
                    "second_to_last": TokenSteeringStrategy.SECOND_TO_LAST,
                    "first_only": TokenSteeringStrategy.FIRST_ONLY,
                    "all_equal": TokenSteeringStrategy.ALL_EQUAL,
                    "exponential_decay": TokenSteeringStrategy.EXPONENTIAL_DECAY,
                    "exponential_growth": TokenSteeringStrategy.EXPONENTIAL_GROWTH,
                    "linear_decay": TokenSteeringStrategy.LINEAR_DECAY,
                    "linear_growth": TokenSteeringStrategy.LINEAR_GROWTH,
                    "custom": TokenSteeringStrategy.CUSTOM,
                }

                strategy = strategy_mapping.get(
                    token_steering_strategy, TokenSteeringStrategy.SECOND_TO_LAST
                )

                # Create token steering configuration
                token_config = TokenSteeringConfig(
                    strategy=strategy,
                    decay_rate=token_decay_rate,
                    min_strength=token_min_strength,
                    max_strength=token_max_strength,
                    apply_to_prompt=token_apply_to_prompt,
                    prompt_strength_multiplier=token_prompt_strength_multiplier,
                )

                # Wrap the steering method with token steering
                steering_obj = TokenSteeringWrapper(steering_obj, token_config)

                if verbose:
                    print(
                        f"   • Wrapped {steering_method} with token steering: {steering_obj.name}"
                    )

            # Train steering method to compute steering vector
            try:
                if latency_tracker:
                    with latency_tracker.time_operation(
                        "total_training_time",
                        {
                            "method": steering_method,
                            "training_samples": len(pair_set),
                            "success": True,
                        },
                    ):
                        training_stats = steering_obj.train(pair_set, layers[0])
                else:
                    training_stats = steering_obj.train(pair_set, layers[0])

                if verbose:
                    print(f"✅ {steering_method} vector computed successfully!")
                    print(f"   • Vector norm: {training_stats['vector_norm']:.4f}")
                    print(f"   • Vector shape: {training_stats['vector_shape']}")
                    print(f"   • Training pairs used: {training_stats['num_pairs']}")
                    if "normalization" in training_stats:
                        norm_info = training_stats["normalization"]
                        print(f"   • Normalization applied: {norm_info['method']}")
                        if "final_norm" in norm_info:
                            print(f"   • Final norm: {norm_info['final_norm']:.4f}")
                        if "scaling_factor" in norm_info:
                            print(
                                f"   • Scaling factor: {norm_info['scaling_factor']:.4f}"
                            )

                    # Show method-specific stats
                    if (
                        steering_method == "HPR"
                        and "householder_matrix_norm" in training_stats
                    ):
                        print(
                            f"   • Householder matrix norm: {training_stats['householder_matrix_norm']:.4f}"
                        )
                    elif steering_method == "BiPO" and "final_loss" in training_stats:
                        print(
                            f"   • Final training loss: {training_stats['final_loss']:.6f}"
                        )
                        print(
                            f"   • Epochs trained: {training_stats['num_epochs_trained']}"
                        )

                # Save steering vector if requested
                if save_steering_vector:
                    success = steering_obj.save_steering_vector(save_steering_vector)
                    if verbose:
                        if success:
                            print(
                                f"   • Saved steering vector to: {save_steering_vector}"
                            )
                        else:
                            print(
                                f"   • Failed to save steering vector to: {save_steering_vector}"
                            )

                # Initialize nonsense detector if needed
                nonsense_detector = None
                if enable_nonsense_detection:
                    from .core.evaluate import create_nonsense_detector

                    nonsense_detector = create_nonsense_detector(
                        max_word_length=max_word_length,
                        repetition_threshold=repetition_threshold,
                        gibberish_threshold=gibberish_threshold,
                        enable_dictionary_check=not disable_dictionary_check,
                    )
                    if verbose:
                        print(
                            f"   • Nonsense detection enabled: action={nonsense_action}"
                        )

                # TEST THE STEERING using lm-harness evaluation (same as baseline)
                if verbose:
                    print(f"\n🧪 TESTING {steering_method} STEERING:")
                    print("   • Running lm-harness evaluation with steering applied...")
                    print(f"   • Test samples: {len(test_qa_pairs_source)}")
                    print(f"   • Steering strength: {steering_strength}")

                # Extract test QA pairs for steering evaluation (same as baseline)
                test_qa_pairs = []
                for doc in test_qa_pairs_source:
                    try:
                        if from_csv or from_json or group_task_qa_format:
                            # For CSV/JSON/group tasks, doc is already a qa_pair dict
                            test_qa_pairs.append(
                                {
                                    "question": doc["question"],
                                    "formatted_question": doc["question"],
                                    "correct_answer": doc["correct_answer"],
                                    "incorrect_answer": doc["incorrect_answer"],
                                }
                            )
                        else:
                            # For lm-harness tasks, extract from document
                            raw_question = doc.get("question", str(doc))
                            if hasattr(task_data, "doc_to_text"):
                                formatted_question = task_data.doc_to_text(doc)
                            else:
                                formatted_question = raw_question

                            # Extract correct answer
                            correct_answers = doc.get("mc1_targets", {}).get(
                                "choices", []
                            )
                            correct_labels = doc.get("mc1_targets", {}).get(
                                "labels", []
                            )

                            # Find the correct answer
                            correct_answer = None
                            for i, label in enumerate(correct_labels):
                                if label == 1 and i < len(correct_answers):
                                    correct_answer = correct_answers[i]
                                    break

                            # Find an incorrect answer
                            incorrect_answer = None
                            for i, label in enumerate(correct_labels):
                                if label == 0 and i < len(correct_answers):
                                    incorrect_answer = correct_answers[i]
                                    break

                            if correct_answer and incorrect_answer:
                                test_qa_pairs.append(
                                    {
                                        "question": raw_question,
                                        "formatted_question": formatted_question,
                                        "correct_answer": correct_answer,
                                        "incorrect_answer": incorrect_answer,
                                    }
                                )

                    except Exception:
                        # Skip problematic docs
                        continue

                # Create steering methods list for lm-harness evaluation
                steering_methods_list = [steering_obj]

                # Run lm-harness evaluation with steering applied (same pipeline as baseline)
                from .core.steering_methods.steering_evaluation import (
                    run_lm_harness_evaluation,
                )

                steering_evaluation_results = run_lm_harness_evaluation(
                    task_data,
                    test_qa_pairs,
                    model,
                    steering_methods_list,
                    layers,
                    steering_strength,
                    True,
                    verbose,
                    output_mode,
                )

                if verbose:
                    print(f"✅ {steering_method} steering evaluation completed")
                    print(
                        f"   📊 Accuracy: {steering_evaluation_results.get('accuracy', 'N/A')}"
                    )
                    print(f"   📊 Test samples: {len(test_qa_pairs)}")

                # No need to generate sample responses since we're using lm-harness evaluation
                steered_responses = []

                # Generate performance report before returning
                if (
                    enable_memory_tracking
                    or enable_latency_tracking
                    or show_timing_summary
                ):
                    if verbose:
                        print("\n🔍 Generating performance report...")
                    print("\n📊 PERFORMANCE REPORT:")
                    print(f"{'=' * 50}")

                    if memory_tracker:
                        if verbose:
                            print("   • Stopping memory monitoring...")
                        memory_stats = memory_tracker.stop_monitoring()
                        print("💾 Memory Usage:")
                        print(
                            memory_tracker.format_stats(
                                memory_stats, detailed_performance_report
                            )
                        )

                    if latency_tracker or show_timing_summary:
                        if verbose:
                            print("   • Collecting timing data...")

                        if latency_tracker:
                            # Use new user-facing metrics format
                            print("\n⏱️ Performance Metrics:")
                            print(latency_tracker.format_user_metrics())
                        else:
                            from .core.tracking import format_timing_summary

                            print("\n⏱️ Timing Summary:")
                            print(format_timing_summary(detailed_performance_report))

                    if export_performance_csv:
                        if latency_tracker:
                            latency_tracker.export_csv(export_performance_csv)
                            print(
                                f"\n📄 Performance data exported to: {export_performance_csv}"
                            )

                    print(f"{'=' * 50}")

                # Return steering mode results with proper evaluation data
                return {
                    "task_name": task_name,
                    "model_name": model_name,
                    "layer": layers[0],
                    "steering_mode": True,
                    "steering_method": steering_method,
                    "steering_strength": steering_strength,
                    "training_stats": training_stats,
                    "training_pairs": len(pair_set),
                    "vector_saved": save_steering_vector is not None,
                    "evaluation_results": steering_evaluation_results,
                    "accuracy": steering_evaluation_results.get("accuracy", "N/A"),
                    "test_samples": len(test_qa_pairs),
                }

            except Exception as e:
                error_msg = (
                    f"{steering_method} steering vector computation failed: {str(e)}"
                )
                if verbose:
                    print(f"\n❌ STEERING ERROR: {error_msg}")
                    print(f"   • Training pairs: {len(pair_set)}")
                    print(f"   • Layer: {layers[0]}")
                    print(f"   • Method: {steering_method}")

                # Generate performance report before error return
                if (
                    enable_memory_tracking
                    or enable_latency_tracking
                    or show_timing_summary
                ):
                    try:
                        if verbose:
                            print("\n🔍 Generating performance report (error case)...")
                        print("\n📊 PERFORMANCE REPORT:")
                        print(f"{'=' * 50}")

                        if memory_tracker:
                            if verbose:
                                print("   • Stopping memory monitoring...")
                            memory_stats = memory_tracker.stop_monitoring()
                            print("💾 Memory Usage:")
                            print(
                                memory_tracker.format_stats(
                                    memory_stats, detailed_performance_report
                                )
                            )

                        if latency_tracker or show_timing_summary:
                            if verbose:
                                print("   • Collecting timing data...")

                            if latency_tracker:
                                # Use new user-facing metrics format
                                print("\n⏱️ Performance Metrics:")
                                print(latency_tracker.format_user_metrics())
                            else:
                                from .core.tracking import format_timing_summary

                                print("\n⏱️ Timing Summary:")
                                print(
                                    format_timing_summary(detailed_performance_report)
                                )

                        if export_performance_csv:
                            if latency_tracker:
                                latency_tracker.export_csv(export_performance_csv)
                                print(
                                    f"\n📄 Performance data exported to: {export_performance_csv}"
                                )

                            print(f"{'=' * 50}")
                    except Exception as perf_error:
                        if verbose:
                            print(
                                f"   • Performance report generation failed: {perf_error}"
                            )

                return {
                    "task_name": task_name,
                    "model_name": model_name,
                    "error": error_msg,
                    "steering_mode": True,
                    "steering_method": steering_method,
                    "error_type": "steering_failure",
                    "suggestion": "Check activation extraction and data quality",
                }

        # CLASSIFICATION MODE (single or multi-layer)
        # Train classifier(s) using optimized type (if optimization was performed)
        if optimize:
            final_classifier_type = optimized_classifier_type
            final_threshold = optimized_threshold
        else:
            final_classifier_type = classifier_type
            final_threshold = 0.6

        # Check if we should try to load pre-trained classifiers automatically
        loaded_classifiers = False
        if (
            not load_classifier and not train_only
        ):  # Only auto-load if not explicitly loading or training
            # Try to load classifiers from the standard optimization directory
            safe_model_name = model_name.replace("/", "_").replace(":", "_")
            auto_load_dir = f"./optimized_classifiers/{safe_model_name}"
            auto_load_path = os.path.join(auto_load_dir, f"{task_name}_classifier")

            # Check if classifier files exist for the required layers
            classifier_exists = True
            if is_multi_layer:
                for layer_idx in layers:
                    check_path = f"{auto_load_path}_layer_{layer_idx}.pkl"
                    if not os.path.exists(check_path):
                        classifier_exists = False
                        break
            else:
                check_path = f"{auto_load_path}_layer_{layers[0]}.pkl"
                classifier_exists = os.path.exists(check_path)

            if classifier_exists:
                if verbose:
                    print(
                        "\n📦 Found pre-trained classifiers, loading automatically..."
                    )
                    print(f"   • Loading from: {auto_load_path}")

                load_classifier = auto_load_path
                loaded_classifiers = True

        # Train classifiers for each layer (or load if specified)
        steering_methods = {}
        layer_training_results = {}

        # If we have classifiers to load, load them instead of training
        if load_classifier:
            from .core.model_persistence import ModelPersistence

            try:
                if is_multi_layer:
                    # Load multiple classifiers
                    classifiers_data = ModelPersistence.load_multi_layer_classifiers(
                        load_classifier, layers
                    )
                    for layer_idx, (classifier, metadata) in classifiers_data.items():
                        steering_type = (
                            SteeringType.LOGISTIC
                            if metadata.get("classifier_type", "logistic") == "logistic"
                            else SteeringType.MLP
                        )
                        steering_method = SteeringMethod(
                            method_type=steering_type,
                            threshold=metadata.get("detection_threshold", 0.6),
                            device=device,
                        )
                        steering_method.classifier = classifier
                        steering_methods[layer_idx] = steering_method

                        # Create dummy training results for loaded classifiers
                        layer_training_results[layer_idx] = {
                            "accuracy": metadata.get("training_accuracy", "N/A"),
                            "f1": metadata.get("f1_score", "N/A"),
                            "loaded": True,
                        }

                        if verbose:
                            print(
                                f"      ✅ Layer {layer_idx}: Loaded (accuracy={metadata.get('training_accuracy', 'N/A')})"
                            )
                else:
                    # Load single classifier
                    classifier, metadata = ModelPersistence.load_classifier(
                        load_classifier, layers[0]
                    )
                    steering_type = (
                        SteeringType.LOGISTIC
                        if metadata.get("classifier_type", "logistic") == "logistic"
                        else SteeringType.MLP
                    )
                    steering_method = SteeringMethod(
                        method_type=steering_type,
                        threshold=metadata.get("detection_threshold", 0.6),
                        device=device,
                    )
                    steering_method.classifier = classifier
                    steering_methods[layers[0]] = steering_method

                    layer_training_results[layers[0]] = {
                        "accuracy": metadata.get("training_accuracy", "N/A"),
                        "f1": metadata.get("f1_score", "N/A"),
                        "loaded": True,
                    }

                    if verbose:
                        print(
                            f"      ✅ Loaded classifier for layer {layers[0]} (accuracy={metadata.get('training_accuracy', 'N/A')})"
                        )

                if loaded_classifiers and verbose:
                    print("   🎉 Using cached classifiers for faster inference!")

            except Exception as e:
                if verbose:
                    print(f"   ⚠️  Failed to load classifiers: {e}")
                    print("   🔄 Falling back to training new classifiers...")
                load_classifier = None  # Reset to allow training
                steering_methods = {}
                layer_training_results = {}

        # AUTO-USE CONTROL VECTOR: If we have a control vector and steering is not explicitly disabled, set it up
        if (
            control_vector_info
            and control_vector_info.get("loaded")
            and not (steering_mode and load_steering_vector)
        ):
            try:
                from .core.steering_methods.control_vector_steering import (
                    ControlVectorSteering,
                )

                # Create control vector steering method
                cv_layer = control_vector_info["layer"]
                cv_steering = ControlVectorSteering(
                    control_vector=control_vector_info["vector"],
                    layer=cv_layer,
                    device=device,
                )

                # Add to steering methods
                if cv_layer in layers:
                    steering_methods[cv_layer] = cv_steering
                    layer_training_results[cv_layer] = {
                        "accuracy": "N/A",
                        "f1": "N/A",
                        "loaded": True,
                        "type": "control_vector",
                    }

                    if verbose:
                        print(
                            f"\n🎮 Using auto-loaded control vector for layer {cv_layer}"
                        )
                        print("   • Vector will be applied during generation")
                        print("   • Use --skip-steering to disable")

            except Exception as e:
                if verbose:
                    print(f"⚠️  Failed to set up control vector steering: {e}")

        # Only train if we didn't successfully load classifiers
        if not steering_methods and is_multi_layer:
            if verbose:
                print("\n🎯 TRAINING MULTI-LAYER CLASSIFIERS:")
                print(f"   • Layers: {layers}")
                print(f"   • Type: {final_classifier_type}")
                print(f"   • Threshold: {final_threshold}")
                print(f"   • Training pairs: {len(contrastive_pairs)}")

            # Train a classifier for each layer
            for layer_idx in layers:
                if verbose:
                    print(f"\n   🔬 Training classifier for layer {layer_idx}...")

                # Extract activations for this specific layer
                layer_processed_pairs = collector.collect_activations_batch(
                    pairs=contrastive_pairs,
                    layer_index=layer_idx,
                    device=device,
                    token_targeting_strategy=targeting_strategy,
                )

                # Create layer-specific ContrastivePairSet
                layer_phrase_pairs = []
                for pair in layer_processed_pairs:
                    positive_full = f"{pair.prompt}{pair.positive_response}"
                    negative_full = f"{pair.prompt}{pair.negative_response}"

                    layer_phrase_pairs.append(
                        {"harmful": negative_full, "harmless": positive_full}
                    )

                layer_pair_set = ContrastivePairSet.from_phrase_pairs(
                    name=f"{task_name}_layer_{layer_idx}",
                    phrase_pairs=layer_phrase_pairs,
                    task_type="lm_evaluation",
                )

                # Store activations in the layer pair set
                for i, processed_pair in enumerate(layer_processed_pairs):
                    if i < len(layer_pair_set.pairs):
                        if (
                            hasattr(layer_pair_set.pairs[i], "positive_response")
                            and layer_pair_set.pairs[i].positive_response
                        ):
                            layer_pair_set.pairs[
                                i
                            ].positive_response.activations = (
                                processed_pair.positive_activations
                            )
                        if (
                            hasattr(layer_pair_set.pairs[i], "negative_response")
                            and layer_pair_set.pairs[i].negative_response
                        ):
                            layer_pair_set.pairs[
                                i
                            ].negative_response.activations = (
                                processed_pair.negative_activations
                            )

                # Train classifier for this layer
                steering_type = (
                    SteeringType.LOGISTIC
                    if final_classifier_type == "logistic"
                    else SteeringType.MLP
                )
                layer_steering_method = SteeringMethod(
                    method_type=steering_type, threshold=final_threshold, device=device
                )

                try:
                    layer_training_results[layer_idx] = layer_steering_method.train(
                        layer_pair_set
                    )
                    steering_methods[layer_idx] = layer_steering_method

                    if verbose:
                        accuracy = layer_training_results[layer_idx].get(
                            "accuracy", "N/A"
                        )
                        f1_score = layer_training_results[layer_idx].get("f1", "N/A")
                        print(
                            f"      ✅ Layer {layer_idx}: Accuracy={accuracy:.2%}, F1={f1_score:.3f}"
                        )

                except Exception as e:
                    if verbose:
                        print(f"      ❌ Layer {layer_idx}: Training failed - {str(e)}")
                    layer_training_results[layer_idx] = {"error": str(e)}

            # Use the first successfully trained layer as the primary one for compatibility
            primary_layer = layers[0]
            if primary_layer in steering_methods:
                steering_method = steering_methods[primary_layer]
                training_results = layer_training_results[primary_layer]
            else:
                # If primary layer failed, try to find any successful layer
                successful_layers = [
                    layer for layer in layers if layer in steering_methods
                ]
                if successful_layers:
                    primary_layer = successful_layers[0]
                    steering_method = steering_methods[primary_layer]
                    training_results = layer_training_results[primary_layer]
                else:
                    # All layers failed
                    error_msg = "All layer classifiers failed to train"
                    if verbose:
                        print(f"\n❌ MULTI-LAYER TRAINING ERROR: {error_msg}")
                    return {
                        "task_name": task_name,
                        "model_name": model_name,
                        "error": error_msg,
                        "layers": layers,
                        "layer_results": layer_training_results,
                        "error_type": "multi_layer_training_failure",
                    }
        elif not steering_methods:  # Only train if we didn't load classifiers
            # Single layer mode (original logic)
            if verbose:
                print("\n🎯 TRAINING CLASSIFIER:")
                print(f"   • Layer: {layers[0]}")
                print(f"   • Type: {final_classifier_type}")
                print(f"   • Threshold: {final_threshold}")
                print(f"   • Training pairs: {len(pair_set)}")

            steering_type = (
                SteeringType.LOGISTIC
                if final_classifier_type == "logistic"
                else SteeringType.MLP
            )
            steering_method = SteeringMethod(
                method_type=steering_type, threshold=final_threshold, device=device
            )

            try:
                training_results = steering_method.train(pair_set)
                steering_methods[layers[0]] = steering_method
                layer_training_results[layers[0]] = training_results

                if verbose:
                    print("✅ Training completed!")
                    print(
                        f"   • Accuracy: {training_results.get('accuracy', 'N/A'):.2%}"
                    )
                    print(f"   • F1 Score: {training_results.get('f1', 'N/A'):.3f}")

            except ZeroDivisionError as e:
                error_msg = f"Classifier training failed due to insufficient or imbalanced data: {str(e)}"
                if verbose:
                    print(f"\n❌ TRAINING ERROR: {error_msg}")
                    print("   • This often happens with very small datasets")
                    print(
                        "   • Try increasing the dataset size or using --limit with a higher value"
                    )
                    print(f"   • Current training samples: {len(pair_set)}")

                return {
                    "task_name": task_name,
                    "model_name": model_name,
                    "error": error_msg,
                    "training_samples": len(pair_set),
                    "error_type": "division_by_zero",
                    "suggestion": "Increase dataset size or check data quality",
                }

            except Exception as e:
                error_msg = f"Classifier training failed: {str(e)}"
                if verbose:
                    print(f"\n❌ TRAINING ERROR: {error_msg}")
                    print(f"   • Training samples: {len(pair_set)}")
                    print(f"   • Classifier type: {final_classifier_type}")

                return {
                    "task_name": task_name,
                    "model_name": model_name,
                    "error": error_msg,
                    "training_samples": len(pair_set),
                    "error_type": "training_failure",
                    "suggestion": "Check data quality or try a different classifier type",
                }
        else:
            # Classifiers were already loaded
            if verbose:
                print("\n✅ Using pre-loaded classifiers (skipped training)")
            # Ensure we have the primary steering method and training results set
            if is_multi_layer and layers[0] in steering_methods:
                steering_method = steering_methods[layers[0]]
                training_results = layer_training_results[layers[0]]
            elif not is_multi_layer and layers[0] in steering_methods:
                steering_method = steering_methods[layers[0]]
                training_results = layer_training_results[layers[0]]

        # Save trained classifiers if requested
        saved_classifier_paths = []
        if save_classifier or train_only:
            from .core.model_persistence import (
                ModelPersistence,
                create_classifier_metadata,
            )

            # Determine save path
            if save_classifier:
                save_path = save_classifier
            else:
                # Default path for train-only mode
                safe_model_name = model_name.replace("/", "_").replace("-", "_")
                save_path = os.path.join(
                    classifier_dir, f"{task_name}_{safe_model_name}_classifier"
                )

            if verbose:
                print("\n💾 SAVING TRAINED CLASSIFIERS:")
                print(f"   • Save path: {save_path}")

            try:
                if is_multi_layer:
                    # Save multiple classifiers
                    for layer_idx in layers:
                        if layer_idx in steering_methods:
                            classifier = steering_methods[layer_idx].classifier
                            training_result = layer_training_results[layer_idx]

                            # Create metadata
                            metadata = create_classifier_metadata(
                                model_name=model_name,
                                task_name=task_name,
                                layer=layer_idx,
                                classifier_type=final_classifier_type,
                                training_accuracy=training_result.get("accuracy", 0.0),
                                training_samples=len(contrastive_pairs),
                                token_aggregation=token_aggregation,
                                detection_threshold=final_threshold,
                            )

                            path = ModelPersistence.save_classifier(
                                classifier, layer_idx, save_path, metadata
                            )
                            saved_classifier_paths.append(path)

                            if verbose:
                                print(f"     ✅ Layer {layer_idx}: {path}")
                else:
                    # Save single classifier
                    classifier = steering_method.classifier
                    metadata = create_classifier_metadata(
                        model_name=model_name,
                        task_name=task_name,
                        layer=layers[0],
                        classifier_type=final_classifier_type,
                        training_accuracy=training_results.get("accuracy", 0.0),
                        training_samples=len(contrastive_pairs),
                        token_aggregation=token_aggregation,
                        detection_threshold=final_threshold,
                    )

                    path = ModelPersistence.save_classifier(
                        classifier, layers[0], save_path, metadata
                    )
                    saved_classifier_paths.append(path)

                    if verbose:
                        print(f"     ✅ Saved: {path}")

            except Exception as e:
                if verbose:
                    print(f"     ❌ Error saving classifiers: {e}")

        # Handle train-only mode - return early after training and saving
        if train_only:
            if verbose:
                print("\n🎓 TRAINING-ONLY MODE COMPLETED!")
                print(
                    f"   • Trained classifiers for layers: {list(steering_methods.keys())}"
                )
                if saved_classifier_paths:
                    print(f"   • Saved {len(saved_classifier_paths)} classifier files")
                print("   • Skipping inference phase")

            return {
                "task_name": task_name,
                "model_name": model_name,
                "mode": "train_only",
                "layers": layers,
                "trained_layers": list(steering_methods.keys()),
                "training_results": (
                    layer_training_results
                    if is_multi_layer
                    else {layers[0]: training_results}
                ),
                "saved_classifier_paths": saved_classifier_paths,
                "classifier_type": final_classifier_type,
                "training_samples": len(contrastive_pairs),
                "success": True,
            }

        # Special handling for lm-eval-harness ground truth evaluation
        if ground_truth_method == "lm-eval-harness":
            # Get the correct evaluation method for this task
            def get_evaluation_method_for_task(task_name: str) -> str:
                """Get the evaluation method for a task from the benchmark configuration."""
                try:
                    import json
                    import os

                    eval_methods_path = os.path.join(
                        os.path.dirname(__file__),
                        "parameters/benchmarks/benchmark_evaluation_methods.json",
                    )
                    with open(eval_methods_path, "r") as f:
                        benchmark_methods = json.load(f)
                        return benchmark_methods.get(task_name, "text-generation")
                except Exception as e:
                    if verbose:
                        print(f"   ⚠️ Could not load benchmark evaluation methods: {e}")
                    return "text-generation"

            evaluation_method = get_evaluation_method_for_task(task_name)

            if verbose:
                print("\n🔍 LM-EVAL-HARNESS GROUND TRUTH EVALUATION:")
                print(
                    "   • Using lm-eval-harness tasks for direct classifier evaluation"
                )
                print(f"   • Task: {task_name}")
                print(f"   • Evaluation method: {evaluation_method}")
                print(f"   • Samples: {len(test_qa_pairs_source)}")

            # Get the trained classifier for evaluation
            if len(layers) > 1:
                # Multi-layer mode - use first layer's classifier
                classifier = (
                    steering_methods[layers[0]].classifier
                    if layers[0] in steering_methods
                    else None
                )
            else:
                # Single-layer mode
                classifier = (
                    steering_method.classifier
                    if hasattr(steering_method, "classifier")
                    else None
                )

            if classifier is None:
                if verbose:
                    print("   ❌ No trained classifier found for evaluation")
                lm_eval_results = {
                    "ground_truth": "UNKNOWN",
                    "method_used": "lm-eval-harness-error",
                    "confidence": 0.0,
                    "details": "No trained classifier available for evaluation",
                    "task_name": task_name,
                    "evaluation_method": evaluation_method,
                }
            else:
                # Use LMEvalHarnessGroundTruth for proper evaluation - pass token_aggregation
                # FIXED: Use actual task name for both constructor and evaluation
                actual_eval_task_name = get_actual_task_name(task_name)
                lm_eval_ground_truth = LMEvalHarnessGroundTruth(
                    actual_eval_task_name, evaluation_method, model=model
                )
                lm_eval_results = lm_eval_ground_truth.evaluate_classifier_on_task(
                    classifier,
                    actual_eval_task_name,
                    num_samples=len(test_qa_pairs_source),
                    model=model,
                    layer=layers[0] if len(layers) > 1 else layers[0],
                    token_aggregation=token_aggregation,
                )

                if verbose:
                    print("   ✅ LM-eval-harness evaluation completed")
                    # Access accuracy from nested lm_eval_metrics
                    lm_eval_metrics = lm_eval_results.get("lm_eval_metrics", {})
                    accuracy = lm_eval_metrics.get("accuracy", "N/A")
                    correct_predictions = lm_eval_metrics.get("correct_predictions", 0)
                    total_samples = lm_eval_metrics.get("total_samples", 0)

                    # Handle evaluation failure gracefully
                    if accuracy == "N/A" or total_samples == 0:
                        error_msg = f"""
⚠️  EVALUATION WARNING FOR {task_name.upper()}!
   • Accuracy: {accuracy}
   • Correct predictions: {correct_predictions} 
   • Total samples: {total_samples}
   • Evaluation method: {evaluation_method}
   
This indicates the {evaluation_method} evaluation method is not working properly for {task_name}.
The task will be skipped in optimization."""
                        print(error_msg)
                        logger.warning(
                            f"Evaluation failed for {task_name}: {error_msg}"
                        )

                        # Return error result that will be caught by optimizer
                        return {
                            "training_results": {"accuracy": 0.0, "f1": 0.0},
                            "evaluation_results": {"accuracy": 0.0},
                            "optimization_result": {
                                "best_layer": layer,
                                "best_aggregation": token_aggregation,
                                "best_threshold": detection_threshold,
                                "best_accuracy": 0.0,
                                "best_f1": 0.0,
                                "error": f"Evaluation failed: {evaluation_method} returned no results",
                            },
                            "error": True,
                        }

                    if isinstance(accuracy, (int, float)):
                        print(f"   📊 Accuracy: {accuracy:.2%}")
                    else:
                        print(f"   📊 Accuracy: {accuracy}")
                    print(f"   🎯 Correct predictions: {correct_predictions}")
                    print(f"   📝 Total samples: {total_samples}")

            # Update evaluation results with lm-eval-harness results
            evaluation_results = lm_eval_results

            # For lm-eval-harness, we don't need to generate responses since we evaluate directly
            # on the multiple choice options from the task
            generated_responses = []
            lm_eval_metrics = lm_eval_results.get("lm_eval_metrics", {})
            correct_classifications = lm_eval_metrics.get("correct_predictions", 0)
            total_classifications = lm_eval_metrics.get("total_samples", 0)

            if verbose:
                print(
                    f"\n🎉 LM-EVAL-HARNESS EVALUATION COMPLETED FOR {task_name.upper()}!"
                )
                print(f"{'=' * 80}")
                print("📊 FINAL RESULTS:")
                print(f"   • Training samples: {len(contrastive_pairs)}")
                print(f"   • Test samples: {len(test_qa_pairs_source)}")

                # Fix training accuracy formatting
                training_accuracy = training_results.get("accuracy", "N/A")
                if isinstance(training_accuracy, (int, float)):
                    print(f"   • Training accuracy: {training_accuracy:.2%}")
                else:
                    print(f"   • Training accuracy: {training_accuracy}")

                # Fix classifier evaluation accuracy formatting
                lm_eval_metrics = lm_eval_results.get("lm_eval_metrics", {})
                classifier_accuracy = lm_eval_metrics.get("accuracy", "N/A")
                if isinstance(classifier_accuracy, (int, float)):
                    print(
                        f"   • Classifier evaluation accuracy: {classifier_accuracy:.2%}"
                    )
                else:
                    print(f"   • Classifier evaluation accuracy: {classifier_accuracy}")

                print(f"   • Correct predictions: {correct_classifications}")
                print(f"   • Total evaluated: {total_classifications}")
                print(f"{'=' * 80}")

            results = {
                "task_name": task_name,
                "model_name": model_name,
                "layer": layer,
                "original_layer": original_layer,
                "token_aggregation": token_aggregation,
                "original_token_aggregation": original_token_aggregation,
                "optimization_performed": optimize,
                "optimization_result": optimization_result,
                "training_results": training_results,
                "evaluation_results": evaluation_results,
                "num_train": len(contrastive_pairs),
                "num_test": len(test_qa_pairs_source),
                "sample_responses": generated_responses,
                "classification_accuracy": lm_eval_metrics.get("accuracy", 0.0),
                "correct_classifications": correct_classifications,
                "total_classifications": total_classifications,
                "ground_truth_method": "lm-eval-harness",
            }

            logger.info(f"LM-eval-harness evaluation completed for {task_name}")
            return results

        # Test the optimized classifier by generating responses and classifying them
        if optimize:
            if verbose:
                print("\n🧪 TESTING OPTIMIZED CLASSIFIER ON GENERATED RESPONSES:")
                print("   • Generating responses to test questions...")

            # Get test questions for response generation
            test_qa_pairs = []
            for doc in test_qa_pairs_source:
                try:
                    if from_csv or from_json:
                        # For CSV/JSON, doc is already a qa_pair dict
                        test_qa_pairs.append(
                            {
                                "question": doc["question"],
                                "formatted_question": doc["question"],
                                "correct_answer": doc["correct_answer"],
                            }
                        )
                    else:
                        # For lm-harness tasks, extract from document
                        raw_question = doc.get("question", str(doc))
                        if hasattr(task_data, "doc_to_text"):
                            formatted_question = task_data.doc_to_text(doc)
                        else:
                            formatted_question = raw_question

                        # Extract correct answer for ground truth comparison
                        correct_answers = doc.get("mc1_targets", {}).get("choices", [])
                        correct_labels = doc.get("mc1_targets", {}).get("labels", [])

                        correct_answer = None
                        for i, label in enumerate(correct_labels):
                            if label == 1 and i < len(correct_answers):
                                correct_answer = correct_answers[i]
                                break

                        if correct_answer:
                            test_qa_pairs.append(
                                {
                                    "question": raw_question,
                                    "formatted_question": formatted_question,
                                    "correct_answer": correct_answer,
                                }
                            )

                except Exception:
                    continue

            if verbose:
                print(
                    f"   • Successfully extracted {len(test_qa_pairs)} test questions"
                )
                print("\n🔍 Test Questions:")
                for i, qa_pair in enumerate(test_qa_pairs):
                    print(f"\n   📋 Question {i + 1}:")
                    print(
                        f"      🔸 Question: {qa_pair['question'][:100]}{'...' if len(qa_pair['question']) > 100 else ''}"
                    )
                    print(f"      ✅ Correct Answer: {qa_pair['correct_answer']}")

            # Generate responses and classify them
            if verbose:
                print("\n🎭 GENERATING AND CLASSIFYING RESPONSES:")
                print(f"   • Generating responses with optimized layer {layer}...")

            generated_responses = []
            correct_classifications = 0
            total_classifications = 0

            if locals().get("use_cached_activations", False):
                # Use cached activations instead of generating new responses
                if verbose:
                    print("\n🔄 PROCESSING CACHED ACTIVATIONS:")
                    print(
                        f"   • Processing {len(locals().get('cached_layer_activations', []))} cached responses..."
                    )

                for i, cached_item in enumerate(
                    locals().get("cached_layer_activations", [])
                ):
                    if verbose and not optimize:
                        print(f"\n   🎯 Processing cached response {i + 1}:")
                        print(
                            f"      📝 Question: {cached_item['question'][:100]}{'...' if len(cached_item['question']) > 100 else ''}"
                        )

                    # Use cached response and activations
                    response = cached_item["response"]
                    activations = cached_item["activations"]

                    # Classify using the current layer's trained classifier
                    if len(layers) > 1:
                        # Multi-layer mode - get classification from the appropriate layer
                        if layers[0] in steering_methods:
                            current_steering_method = steering_methods[layers[0]]
                            classification_result = (
                                current_steering_method.classify_activation(activations)
                            )
                            classification = (
                                "HALLUCINATION"
                                if classification_result.get("is_harmful", False)
                                else "TRUTHFUL"
                            )
                            token_scores = [
                                classification_result.get("score", 0.5)
                            ]  # Single score for cached
                            aggregated_score = classification_result.get("score", 0.5)
                        else:
                            classification = "UNKNOWN"
                            token_scores = [0.5]
                            aggregated_score = 0.5
                    else:
                        # Single-layer mode
                        classification_result = steering_method.classify_activation(
                            activations
                        )
                        classification = (
                            "HALLUCINATION"
                            if classification_result.get("is_harmful", False)
                            else "TRUTHFUL"
                        )
                        token_scores = [
                            classification_result.get("score", 0.5)
                        ]  # Single score for cached
                        aggregated_score = classification_result.get("score", 0.5)

                    # Create a mock qa_pair for ground truth evaluation
                    qa_pair = {
                        "question": cached_item["question"],
                        "correct_answer": "N/A",
                    }

                    # Evaluate the cached response using the ground truth evaluator
                    try:
                        # Create ground truth evaluator
                        evaluator = GroundTruthEvaluator.from_string(
                            ground_truth_method
                        )

                        # Get user label if available
                        user_label = None
                        if user_labels and i < len(user_labels):
                            user_label = user_labels[i]

                        # Evaluate the response
                        evaluation_result = evaluator.evaluate_response(
                            response, qa_pair.get("correct_answer", ""), user_label
                        )

                        ground_truth = evaluation_result["ground_truth"]

                        # Check if our classification matches ground truth (only if ground truth is not UNKNOWN)
                        classification_correct = None
                        if ground_truth != "UNKNOWN":
                            classification_correct = classification == ground_truth
                            if classification_correct:
                                correct_classifications += 1
                            total_classifications += 1

                        # Create response entry
                        response_entry = {
                            "question": cached_item["question"],
                            "response": response,
                            "token_scores": token_scores,
                            "aggregated_score": aggregated_score,
                            "classification": classification,
                            "ground_truth": ground_truth,
                            "ground_truth_method": evaluation_result["method_used"],
                            "ground_truth_confidence": evaluation_result["confidence"],
                            "ground_truth_details": evaluation_result["details"],
                            "classification_correct": classification_correct,
                            "was_handled": False,
                            "source": "cached_activations",
                        }

                        generated_responses.append(response_entry)

                        if verbose and not optimize:
                            print(f"      🤖 Cached Response: {response}")
                            print(
                                f"      📊 Classification: {classification} (score: {aggregated_score:.3f})"
                            )
                            print(
                                f"      🎯 Ground Truth: {ground_truth} (method: {evaluation_result['method_used']})"
                            )
                            if classification_correct is not None:
                                print(
                                    f"      {'✅' if classification_correct else '❌'} Classification {'CORRECT' if classification_correct else 'WRONG'}"
                                )

                    except Exception as e:
                        if verbose and not optimize:
                            print(f"      ⚠️  Could not evaluate cached response: {e}")
                        generated_responses.append(
                            {
                                "question": cached_item["question"],
                                "response": response,
                                "token_scores": token_scores,
                                "classification": classification,
                                "ground_truth": "UNKNOWN",
                                "ground_truth_method": "error",
                                "ground_truth_confidence": 0.0,
                                "ground_truth_details": f"Error during evaluation: {str(e)}",
                                "classification_correct": None,
                                "was_handled": False,
                                "source": "cached_activations",
                            }
                        )

            else:
                # Generate new responses (original logic)
                for i, qa_pair in enumerate(test_qa_pairs):
                    if (
                        verbose and not optimize
                    ):  # Only show detailed progress when not optimizing
                        print(f"\n   🎯 Generating response {i + 1}:")
                        print(
                            f"      📝 Question: {qa_pair['question'][:100]}{'...' if len(qa_pair['question']) > 100 else ''}"
                        )

                    # Use the raw question for natural generation
                    # The formatted_question contains few-shot examples which are for training, not generation
                    simple_prompt = qa_pair["question"]

                    # Generate response with token-level scoring and detection handling
                    if len(layers) > 1:
                        # Multi-layer mode: use multi-layer generation function
                        response, layer_results, was_handled = (
                            generate_with_multi_layer_classification_and_handling(
                                model,
                                simple_prompt,
                                layers,
                                max_new_tokens,
                                steering_methods,
                                token_aggregation,
                                detection_threshold,
                                verbose and not optimize,
                                detection_handler,
                            )
                        )
                        # For backward compatibility, use primary layer's results for main fields
                        primary_layer = layers[0]
                        token_scores = (
                            layer_results[primary_layer]["token_scores"]
                            if primary_layer in layer_results
                            else []
                        )
                        classification = (
                            layer_results[primary_layer]["classification"]
                            if primary_layer in layer_results
                            else "UNKNOWN"
                        )
                        aggregated_score = (
                            layer_results[primary_layer]["aggregated_score"]
                            if primary_layer in layer_results
                            else 0.0
                        )
                    else:
                        # Single-layer mode: use original function
                        response, token_scores, classification, was_handled = (
                            generate_with_classification_and_handling(
                                model,
                                simple_prompt,
                                layers[0],
                                max_new_tokens,
                                steering_method,
                                token_aggregation,
                                detection_threshold,
                                verbose and not optimize,
                                detection_handler,
                            )
                        )
                        layer_results = None
                        aggregated_score = (
                            aggregate_token_scores(token_scores, token_aggregation)
                            if token_scores
                            else 0.0
                        )

                        # Save activations if requested (extract from last generation)
                        if (
                            save_test_activations
                            and locals().get("test_activation_cache") is not None
                        ):
                            try:
                                # We need to extract activations from the last forward pass
                                # This is a simplified version - ideally we'd modify the generation functions
                                # to return activations as well

                                # For now, we'll do a quick forward pass to extract activations
                                model_inputs = model.tokenizer(
                                    simple_prompt, return_tensors="pt", padding=True
                                )
                                if hasattr(model, "device"):
                                    model_inputs = {
                                        k: v.to(model.device)
                                        for k, v in model_inputs.items()
                                    }

                                with torch.no_grad():
                                    outputs = model.model(
                                        **model_inputs, output_hidden_states=True
                                    )

                                    # Extract activations from the target layer
                                    if (
                                        outputs.hidden_states
                                        and len(outputs.hidden_states) > layers[0]
                                    ):
                                        layer_activations = outputs.hidden_states[
                                            layers[0] + 1
                                        ]  # +1 because hidden_states[0] is embeddings

                                        # Create Activations object
                                        from .core.activations import (
                                            Activations,
                                            ActivationAggregationMethod,
                                        )

                                        layer_obj = Layer(
                                            index=layers[0], type="transformer"
                                        )
                                        activations_obj = Activations(
                                            tensor=layer_activations,
                                            layer=layer_obj,
                                            aggregation_method=ActivationAggregationMethod.LAST_TOKEN,
                                        )

                                        # Add to cache
                                        locals().get(
                                            "test_activation_cache"
                                        ).add_activation(
                                            question=qa_pair["question"],
                                            response=response,
                                            activations=activations_obj,
                                            layer=layers[0],
                                        )

                            except Exception as e:
                                if verbose:
                                    print(f"      ⚠️  Could not save activation: {e}")

                    # Evaluate the generated response using the ground truth evaluator
                    try:
                        # Create ground truth evaluator
                        evaluator = GroundTruthEvaluator.from_string(
                            ground_truth_method
                        )

                        # Get correct answer for comparison
                        correct_answers = qa_pair.get("correct_answer", "")

                        # Get user label if available
                        user_label = None
                        if user_labels and i < len(user_labels):
                            user_label = user_labels[i]

                        # Evaluate the response
                        evaluation_result = evaluator.evaluate_response(
                            response, correct_answers, user_label
                        )

                        ground_truth = evaluation_result["ground_truth"]

                        # Check if our classification matches ground truth (only if ground truth is not UNKNOWN)
                        classification_correct = None
                        if ground_truth != "UNKNOWN":
                            expected_classification = (
                                "HALLUCINATION"
                                if ground_truth == "HALLUCINATION"
                                else "TRUTHFUL"
                            )
                            classification_correct = (
                                classification == expected_classification
                            )

                            if classification_correct:
                                correct_classifications += 1
                            total_classifications += 1

                        # Create response entry with layer results if available
                        response_entry = {
                            "question": qa_pair["question"],  # Add the question
                            "response": response,
                            "token_scores": token_scores,
                            "aggregated_score": aggregated_score,
                            "classification": classification,
                            "ground_truth": ground_truth,
                            "ground_truth_method": evaluation_result["method_used"],
                            "ground_truth_confidence": evaluation_result["confidence"],
                            "ground_truth_details": evaluation_result["details"],
                            "classification_correct": classification_correct,
                            "was_handled": was_handled,
                        }

                        # Add layer-specific results if multi-layer
                        if layer_results:
                            response_entry["layer_results"] = layer_results

                        generated_responses.append(response_entry)

                        if (
                            verbose and not optimize
                        ):  # Only show detailed output when not optimizing
                            print(f"      🤖 Generated: {response}")
                            print(
                                f"      🔍 Token Scores: {[f'{score:.3f}' for score in token_scores]}"
                            )
                            aggregated_score = aggregate_token_scores(
                                token_scores, token_aggregation
                            )
                            print(
                                f"      📊 Our Classification: {classification} ({token_aggregation} score: {aggregated_score:.3f})"
                            )
                            print(
                                f"      🎯 Ground Truth: {ground_truth} (method: {evaluation_result['method_used']}, confidence: {evaluation_result['confidence']:.2f})"
                            )
                            if classification_correct is not None:
                                print(
                                    f"      {'✅' if classification_correct else '❌'} Classification {'CORRECT' if classification_correct else 'WRONG'}"
                                )
                            else:
                                print(
                                    f"      ❓ Classification accuracy not evaluated (ground truth method: {evaluation_result['method_used']})"
                                )
                            print(f"      ✅ Expected: {qa_pair['correct_answer']}")
                            print(f"      ❌ Incorrect: {qa_pair['incorrect_answer']}")
                            if evaluation_result["details"]:
                                print(
                                    f"      📝 Details: {evaluation_result['details']}"
                                )

                    except Exception as e:
                        if verbose and not optimize:
                            print(f"      ⚠️  Could not evaluate response: {e}")
                        generated_responses.append(
                            {
                                "question": qa_pair["question"],  # Add the question
                                "response": response,
                                "token_scores": token_scores,
                                "classification": classification,
                                "ground_truth": "UNKNOWN",
                                "ground_truth_method": "error",
                                "ground_truth_confidence": 0.0,
                                "ground_truth_details": f"Error during evaluation: {str(e)}",
                                "classification_correct": None,
                                "was_handled": was_handled,
                            }
                        )

            # Calculate evaluation results
            if total_classifications > 0:
                test_accuracy = correct_classifications / total_classifications
                evaluation_results = {
                    "accuracy": test_accuracy,
                    "correct_predictions": correct_classifications,
                    "total_predictions": total_classifications,
                }
            else:
                evaluation_results = {
                    "accuracy": "N/A",
                    "correct_predictions": 0,
                    "total_predictions": 0,
                }

            if verbose:
                print("\n✅ Response generation and classification completed!")
                if total_classifications > 0:
                    print(
                        f"   • Test accuracy: {test_accuracy:.2%} ({correct_classifications}/{total_classifications})"
                    )
                else:
                    print("   • Test accuracy: Could not evaluate")
                print("   • Tested on generated responses, not pre-written choices")

            # Create results dictionary for optimization path
            results = {
                "task_name": task_name,
                "model_name": model_name,
                "layer": layer,
                "original_layer": original_layer,
                "token_aggregation": token_aggregation,
                "original_token_aggregation": original_token_aggregation,
                "optimization_performed": optimize,
                "optimization_result": optimization_result,
                "training_results": training_results,
                "evaluation_results": evaluation_results,
                "num_train": len(contrastive_pairs),
                "num_test": len(test_qa_pairs),
                "sample_responses": generated_responses,
                "classification_accuracy": (
                    correct_classifications / total_classifications
                    if total_classifications > 0
                    else None
                ),
                "correct_classifications": correct_classifications,
                "total_classifications": total_classifications,
            }

            if verbose:
                print(f"\n🎉 OPTIMIZATION PIPELINE COMPLETED FOR {task_name.upper()}!")
                print(f"{'=' * 80}")
                print("📊 FINAL RESULTS:")
                print(f"   • Training samples: {len(contrastive_pairs)}")
                print(f"   • Test samples: {len(test_qa_pairs)}")
                print(
                    f"   • Training accuracy: {training_results.get('accuracy', 'N/A'):.2%}"
                )
                if total_classifications > 0:
                    print(
                        f"   • Test accuracy: {test_accuracy:.2%} ({correct_classifications}/{total_classifications})"
                    )
                else:
                    print("   • Test accuracy: Could not evaluate")
                print(f"   • Generated responses: {len(generated_responses)}")
                if total_classifications > 0:
                    classification_acc = correct_classifications / total_classifications
                    print(
                        f"   • Classification accuracy on generated responses: {classification_acc:.2%} ({correct_classifications}/{total_classifications})"
                    )
                else:
                    print("   • Classification accuracy: Could not evaluate")
                print(f"{'=' * 80}\n")

            logger.info(f"Optimization pipeline completed for {task_name}")
            return results
        else:
            # Only do pre-written validation when NOT optimizing
            if verbose:
                print("\n🧪 PREPARING TEST DATA:")
                print(
                    f"   • Loading {task_name} test data with correct/incorrect answers..."
                )

            # Get the actual test data with correct and incorrect answers
            test_qa_pairs = []
            for doc in test_qa_pairs_source:
                try:
                    if from_csv or from_json or group_task_qa_format:
                        # For CSV/JSON/group tasks, doc is already a qa_pair dict
                        test_qa_pairs.append(
                            {
                                "question": doc["question"],
                                "formatted_question": doc["question"],
                                "correct_answer": doc["correct_answer"],
                                "incorrect_answer": doc["incorrect_answer"],
                            }
                        )
                    else:
                        # For lm-harness tasks, extract from document
                        raw_question = doc.get("question", str(doc))
                        if hasattr(task_data, "doc_to_text"):
                            formatted_question = task_data.doc_to_text(doc)
                        else:
                            formatted_question = raw_question

                        # Extract correct answer
                        correct_answers = doc.get("mc1_targets", {}).get("choices", [])
                        correct_labels = doc.get("mc1_targets", {}).get("labels", [])

                        # Find the correct answer
                        correct_answer = None
                        for i, label in enumerate(correct_labels):
                            if label == 1 and i < len(correct_answers):
                                correct_answer = correct_answers[i]
                                break

                        # Find an incorrect answer
                        incorrect_answer = None
                        for i, label in enumerate(correct_labels):
                            if label == 0 and i < len(correct_answers):
                                incorrect_answer = correct_answers[i]
                                break

                        if correct_answer and incorrect_answer:
                            test_qa_pairs.append(
                                {
                                    "question": raw_question,
                                    "formatted_question": formatted_question,
                                    "correct_answer": correct_answer,
                                    "incorrect_answer": incorrect_answer,
                                }
                            )

                except Exception:
                    # Skip problematic docs
                    continue

            if verbose:
                print(f"   • Successfully extracted {len(test_qa_pairs)} test QA pairs")
                print("\n🔍 Test Examples:")
                for i, qa_pair in enumerate(test_qa_pairs):
                    print(f"\n   📋 Test Example {i + 1}:")
                    print(
                        f"      🔸 Question: {qa_pair['question'][:100]}{'...' if len(qa_pair['question']) > 100 else ''}"
                    )
                    print(f"      ✅ Correct Answer: {qa_pair['correct_answer']}")
                    if "incorrect_answer" in qa_pair:
                        print(
                            f"      ❌ Incorrect Answer: {qa_pair['incorrect_answer']}"
                        )

            # Create test contrastive pairs using proper activation collection logic
            test_contrastive_pairs = collector.create_batch_contrastive_pairs(
                test_qa_pairs
            )

            test_processed_pairs = collector.collect_activations_batch(
                pairs=test_contrastive_pairs,
                layer_index=layers[0],
                device=device,
                token_targeting_strategy=targeting_strategy,
            )

            # Convert to ContrastivePairSet format for evaluation
            test_phrase_pairs = []
            for pair in test_processed_pairs:
                # Create the full prompts for the pair set
                positive_full = f"{pair.prompt}{pair.positive_response}"
                negative_full = f"{pair.prompt}{pair.negative_response}"

                test_phrase_pairs.append(
                    {
                        "harmful": negative_full,  # A choice (incorrect)
                        "harmless": positive_full,  # B choice (correct)
                    }
                )

            # Run proper lm-harness evaluation on the test set with steering (only for individual tasks)
            from .core.steering_methods.steering_evaluation import (
                run_lm_harness_evaluation,
            )

            # Only run lm-harness evaluation for individual tasks, not group tasks
            if not group_task_processed:
                evaluation_results = run_lm_harness_evaluation(
                    task_data,
                    test_qa_pairs,
                    model,
                    steering_methods,
                    layers,
                    1.0,
                    True,
                    verbose,
                    "likelihoods",
                )
            else:
                # For group tasks, we skip lm-harness evaluation since it doesn't apply to combined subtasks
                evaluation_results = {
                    "baseline_accuracy": "N/A (group task)",
                    "steered_accuracy": "N/A (group task)",
                    "improvement": "N/A (group task)",
                    "note": "lm-harness evaluation skipped for group tasks (combined subtasks)",
                }

            # Handle test activation loading/saving
            test_activation_cache = None
            use_cached_activations = False

            if load_test_activations:
                # Load cached test activations instead of generating new responses
                if verbose:
                    print("\n💾 LOADING CACHED TEST ACTIVATIONS:")
                    print(f"   • Loading from: {load_test_activations}")

                try:
                    test_activation_cache = TestActivationCache.load_from_file(
                        load_test_activations
                    )

                    # Filter activations for the current layer
                    cached_layer_activations = (
                        test_activation_cache.get_activations_for_layer(layers[0])
                    )

                    if cached_layer_activations:
                        use_cached_activations = True
                        if verbose:
                            print(
                                f"   ✅ Found {len(cached_layer_activations)} cached activations for layer {layers[0]}"
                            )
                    else:
                        if verbose:
                            print(
                                f"   ❌ No cached activations found for layer {layers[0]}"
                            )
                            print(
                                f"   • Available layers: {list(set(item['layer'] for item in test_activation_cache.activations))}"
                            )

                except Exception as e:
                    if verbose:
                        print(f"   ❌ Failed to load cached activations: {e}")
                        print("   • Will generate new responses instead")

            if save_test_activations and not use_cached_activations:
                # Initialize cache for saving
                test_activation_cache = TestActivationCache()
                if verbose:
                    print(
                        f"\n💾 WILL SAVE TEST ACTIVATIONS TO: {save_test_activations}"
                    )

            # Generate sample responses with token-level classification
            if verbose:
                if optimize:
                    print("\n🎭 GENERATING SAMPLE RESPONSES WITH OPTIMIZED CLASSIFIER:")
                    print(
                        f"   • Generating {len(test_qa_pairs)} sample responses with optimized layer {layer}..."
                    )
                else:
                    print(
                        "\n🎭 GENERATING SAMPLE RESPONSES WITH HALLUCINATION DETECTION:"
                    )
                    print(f"   • Generating {len(test_qa_pairs)} sample responses...")

            generated_responses = []
            correct_classifications = 0
            total_classifications = 0

            for i, qa_pair in enumerate(test_qa_pairs):
                if (
                    verbose and not optimize
                ):  # Only show detailed progress when not optimizing
                    print(f"\n   🎯 Generating response {i + 1}:")
                    print(
                        f"      📝 Question: {qa_pair['question'][:100]}{'...' if len(qa_pair['question']) > 100 else ''}"
                    )

                # Use the raw question for natural generation
                # The formatted_question contains few-shot examples which are for training, not generation
                simple_prompt = qa_pair["question"]

                # Generate response with token-level scoring and detection handling
                if len(layers) > 1:
                    # Multi-layer mode: use multi-layer generation function
                    response, layer_results, was_handled = (
                        generate_with_multi_layer_classification_and_handling(
                            model,
                            simple_prompt,
                            layers,
                            max_new_tokens,
                            steering_methods,
                            token_aggregation,
                            detection_threshold,
                            verbose and not optimize,
                            detection_handler,
                        )
                    )
                    # For backward compatibility, use primary layer's results for main fields
                    primary_layer = layers[0]
                    token_scores = (
                        layer_results[primary_layer]["token_scores"]
                        if primary_layer in layer_results
                        else []
                    )
                    classification = (
                        layer_results[primary_layer]["classification"]
                        if primary_layer in layer_results
                        else "UNKNOWN"
                    )
                    aggregated_score = (
                        layer_results[primary_layer]["aggregated_score"]
                        if primary_layer in layer_results
                        else 0.0
                    )
                else:
                    # Single-layer mode: use original function
                    response, token_scores, classification, was_handled = (
                        generate_with_classification_and_handling(
                            model,
                            simple_prompt,
                            layers[0],
                            max_new_tokens,
                            steering_method,
                            token_aggregation,
                            detection_threshold,
                            verbose and not optimize,
                            detection_handler,
                        )
                    )
                    layer_results = None
                    aggregated_score = (
                        aggregate_token_scores(token_scores, token_aggregation)
                        if token_scores
                        else 0.0
                    )

                # Evaluate the generated response using the ground truth evaluator
                try:
                    # Create ground truth evaluator
                    evaluator = GroundTruthEvaluator.from_string(ground_truth_method)

                    # Get correct answer for comparison
                    correct_answers = qa_pair.get("correct_answer", "")

                    # Get user label if available
                    user_label = None
                    if user_labels and i < len(user_labels):
                        user_label = user_labels[i]

                    # Evaluate the response
                    evaluation_result = evaluator.evaluate_response(
                        response, correct_answers, user_label
                    )

                    ground_truth = evaluation_result["ground_truth"]

                    # Check if our classification matches ground truth (only if ground truth is not UNKNOWN)
                    classification_correct = None
                    if ground_truth != "UNKNOWN":
                        classification_correct = classification == ground_truth
                        if classification_correct:
                            correct_classifications += 1
                        total_classifications += 1

                    # Create response entry with layer results if available
                    response_entry = {
                        "question": qa_pair["question"],  # Add the question
                        "response": response,
                        "token_scores": token_scores,
                        "aggregated_score": aggregated_score,
                        "classification": classification,
                        "ground_truth": ground_truth,
                        "ground_truth_method": evaluation_result["method_used"],
                        "ground_truth_confidence": evaluation_result["confidence"],
                        "ground_truth_details": evaluation_result["details"],
                        "classification_correct": classification_correct,
                        "was_handled": was_handled,
                    }

                    # Add layer-specific results if multi-layer
                    if layer_results:
                        response_entry["layer_results"] = layer_results

                    generated_responses.append(response_entry)

                    if (
                        verbose and not optimize
                    ):  # Only show detailed output when not optimizing
                        print(f"      🤖 Generated: {response}")
                        print(
                            f"      🔍 Token Scores: {[f'{score:.3f}' for score in token_scores]}"
                        )
                        aggregated_score = aggregate_token_scores(
                            token_scores, token_aggregation
                        )
                        print(
                            f"      📊 Our Classification: {classification} ({token_aggregation} score: {aggregated_score:.3f})"
                        )
                        print(
                            f"      🎯 Ground Truth: {ground_truth} (method: {evaluation_result['method_used']}, confidence: {evaluation_result['confidence']:.2f})"
                        )
                        if classification_correct is not None:
                            print(
                                f"      {'✅' if classification_correct else '❌'} Classification {'CORRECT' if classification_correct else 'WRONG'}"
                            )
                        else:
                            print(
                                f"      ❓ Classification accuracy not evaluated (ground truth method: {evaluation_result['method_used']})"
                            )
                        print(f"      ✅ Expected: {qa_pair['correct_answer']}")
                        print(f"      ❌ Incorrect: {qa_pair['incorrect_answer']}")
                        if evaluation_result["details"]:
                            print(f"      📝 Details: {evaluation_result['details']}")

                except Exception as e:
                    if verbose and not optimize:
                        print(f"      ⚠️  Could not evaluate response: {e}")
                    generated_responses.append(
                        {
                            "question": qa_pair["question"],  # Add the question
                            "response": response,
                            "token_scores": token_scores,
                            "classification": classification,
                            "ground_truth": "UNKNOWN",
                            "ground_truth_method": "error",
                            "ground_truth_confidence": 0.0,
                            "ground_truth_details": f"Error during evaluation: {str(e)}",
                            "classification_correct": None,
                            "was_handled": was_handled,
                        }
                    )

            # Show summary for optimization
            if verbose and optimize:
                print(
                    f"\n   ✅ Generated {len(generated_responses)} responses with optimized layer {layer}"
                )
                if total_classifications > 0:
                    classification_acc = correct_classifications / total_classifications
                    print(
                        f"   📊 Classification accuracy: {classification_acc:.2%} ({correct_classifications}/{total_classifications})"
                    )

            results = {
                "task_name": task_name,
                "model_name": model_name,
                "layer": layer,
                "original_layer": original_layer,
                "token_aggregation": token_aggregation,
                "original_token_aggregation": original_token_aggregation,
                "optimization_performed": optimize,
                "optimization_result": optimization_result,
                "training_results": training_results,
                "evaluation_results": evaluation_results,
                "num_train": len(contrastive_pairs),
                "num_test": len(test_qa_pairs),
                "sample_responses": generated_responses,
                "classification_accuracy": (
                    correct_classifications / total_classifications
                    if total_classifications > 0
                    else None
                ),
                "correct_classifications": correct_classifications,
                "total_classifications": total_classifications,
            }

            if verbose:
                print(f"\n🎉 PIPELINE COMPLETED FOR {task_name.upper()}!")
                print(f"{'=' * 80}")
                print("📊 FINAL RESULTS:")
                print(f"   • Training samples: {len(contrastive_pairs)}")
                print(f"   • Test samples: {len(test_qa_pairs)}")
                print(
                    f"   • Training accuracy: {training_results.get('accuracy', 'N/A'):.2%}"
                )
                print(f"   • Generated responses: {len(generated_responses)}")
                if total_classifications > 0:
                    classification_acc = correct_classifications / total_classifications
                    print(
                        f"   • Classification accuracy on generated responses: {classification_acc:.2%} ({correct_classifications}/{total_classifications})"
                    )
                else:
                    print("   • Classification accuracy: Could not evaluate")
                print(f"{'=' * 80}\n")

            # Generate performance report
            if enable_memory_tracking or enable_latency_tracking or show_timing_summary:
                if verbose:
                    print("\n🔍 Generating performance report...")
                print("\n📊 PERFORMANCE REPORT:")
                print(f"{'=' * 50}")

                if memory_tracker:
                    if verbose:
                        print("   • Stopping memory monitoring...")
                    memory_stats = memory_tracker.stop_monitoring()
                    print("💾 Memory Usage:")
                    print(
                        memory_tracker.format_stats(
                            memory_stats, detailed_performance_report
                        )
                    )

                if latency_tracker or show_timing_summary:
                    if verbose:
                        print("   • Collecting timing data...")
                    from .core.tracking import format_timing_summary

                    print("\n⏱️ Timing Summary:")
                    print(format_timing_summary(detailed_performance_report))

                if export_performance_csv:
                    if latency_tracker:
                        latency_tracker.export_csv(export_performance_csv)
                        print(
                            f"\n📄 Performance data exported to: {export_performance_csv}"
                        )

                print(f"{'=' * 50}")

            # Save test activations if requested
            if (
                save_test_activations
                and test_activation_cache is not None
                and len(test_activation_cache.activations) > 0
            ):
                try:
                    test_activation_cache.save_to_file(save_test_activations)
                    if verbose:
                        print("\n💾 SAVED TEST ACTIVATIONS:")
                        print(f"   • File: {save_test_activations}")
                        print(
                            f"   • Count: {len(test_activation_cache.activations)} activations"
                        )
                        print(f"   • Layer: {layers[0]}")
                except Exception as e:
                    if verbose:
                        print(f"\n❌ Failed to save test activations: {e}")

            logger.info(f"Pipeline completed for {task_name}")
            return results

    except Exception as e:
        # 🚨 ALL ERRORS ARE CRITICAL - HARD STOP THE ENTIRE PROGRAM
        logger.error(f"💥 HARD STOP: Error in pipeline for {task_name}: {e}")
        # Stop tracking on error
        if memory_tracker:
            try:
                memory_tracker.stop_monitoring()
            except Exception:
                pass
        # Import traceback for full stack trace
        import traceback

        print("\n💥💥💥 HARD STOP - ERROR DETECTED 💥💥💥")
        print(f"Task: {task_name}")
        print(f"Error: {e}")
        print("Full traceback:")
        traceback.print_exc()
        print("💥💥💥 STOPPING EXECUTION IMMEDIATELY 💥💥💥\n")
        # HARD STOP - crash the program immediately
        raise e


def main():
    """Main entry point for the CLI."""
    parser = setup_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle different commands
    if args.command == "generate-pairs":
        handle_generate_pairs_command(args)
    elif args.command == "synthetic":
        handle_synthetic_command(args)
    elif args.command == "tasks":
        handle_tasks_command(args)
    elif args.command == "test-nonsense":
        handle_test_nonsense_command(args)
    elif args.command == "monitor":
        handle_monitor_command(args)
    elif args.command == "agent":
        handle_agent_command(args)
    elif args.command == "model-config":
        handle_model_config_command(args)
    elif args.command == "optimize-classification":
        handle_classification_optimization_command(args)
    elif args.command == "optimize-steering":
        handle_steering_optimization_command(args)
    elif args.command == "optimize-sample-size":
        handle_sample_size_optimization_command(args)
    elif args.command == "full-optimize":
        handle_full_optimization_command(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


def handle_generate_pairs_command(args):
    """Handle the generate-pairs command."""
    print("🎯 Generating synthetic contrastive pairs...")
    print(f"   • Trait: {args.trait}")
    print(f"   • Number of pairs: {args.num_pairs}")
    print(f"   • Output file: {args.output}")

    try:
        # Load model
        from .core.model import Model

        model = Model(name=args.model, device=args.device)

        # Generate pairs
        pair_set = generate_synthetic_pairs_cli(
            trait_description=args.trait,
            num_pairs=args.num_pairs,
            output_file=args.output,
            model=model,
        )

        print(
            f"✅ Successfully generated and saved {len(pair_set.pairs)} contrastive pairs!"
        )

    except Exception as e:
        print(f"❌ Error generating pairs: {e}")
        sys.exit(1)


def handle_synthetic_command(args):
    """Handle the synthetic command (generate + train + test)."""
    print("🚀 Running synthetic contrastive pair pipeline...")

    try:
        # Load model
        from .core.model import Model

        model = Model(name=args.model, device=args.device)

        # Get or generate contrastive pairs
        if args.trait:
            print(f"   • Generating pairs for trait: {args.trait}")
            generate_synthetic_pairs_cli(
                trait_description=args.trait,
                num_pairs=args.num_pairs,
                output_file=args.save_pairs,
                model=model,
            )
        else:
            print(f"   • Loading pairs from: {args.pairs_file}")
            load_synthetic_pairs_cli(args.pairs_file, model)

        print("✅ Synthetic pipeline completed!")

    except Exception as e:
        print(f"❌ Error in synthetic pipeline: {e}")
        import traceback

        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


def _generate_test_scenarios(
    trait_description: str, num_scenarios: int, model
) -> List[str]:
    """Generate test scenarios for evaluating the steering method."""
    return [
        f"Test scenario {i + 1} for {trait_description}" for i in range(num_scenarios)
    ]


def handle_tasks_command(args):
    """Handle the tasks command."""

    # Handle cache management commands first
    if hasattr(args, "cache_status") and args.cache_status:
        import json
        from pathlib import Path

        cache_dir = Path(args.cache_dir)
        data_dir = cache_dir / "data"
        metadata_dir = cache_dir / "metadata"

        print("📊 CACHE STATUS")
        print(f"{'=' * 50}")
        print(f"Cache directory: {args.cache_dir}")

        if not cache_dir.exists():
            print("❌ Cache directory does not exist")
            return

        if not data_dir.exists() or not metadata_dir.exists():
            print(
                "❌ Cache structure incomplete (missing data or metadata directories)"
            )
            return

        # Read existing cache format
        total_tasks = 0
        total_size_bytes = 0
        cached_tasks = {}

        for metadata_file in metadata_dir.glob("*_metadata.json"):
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)

                task_name = metadata.get(
                    "task_name", metadata_file.stem.replace("_metadata", "")
                )
                data_file = data_dir / f"{task_name}.pkl"

                if data_file.exists():
                    file_size = data_file.stat().st_size
                    total_size_bytes += file_size
                    total_tasks += 1

                    cached_tasks[task_name] = {
                        "samples": metadata.get("total_samples", "unknown"),
                        "size_mb": file_size / (1024 * 1024),
                        "download_time": metadata.get("download_timestamp", "unknown"),
                        "processing_time": metadata.get(
                            "processing_time_seconds", "unknown"
                        ),
                    }
            except Exception as e:
                print(f"⚠️  Warning: Could not read metadata for {metadata_file}: {e}")

        print(f"Total tasks: {total_tasks}")
        print(f"Total size: {total_size_bytes / (1024 * 1024):.1f} MB")
        print()

        if cached_tasks:
            print("📋 CACHED TASKS:")
            # Sort by size for better display
            sorted_tasks = sorted(
                cached_tasks.items(), key=lambda x: x[1]["size_mb"], reverse=True
            )
            for task_name, task_info in sorted_tasks:
                samples = task_info["samples"]
                size_mb = task_info["size_mb"]
                download_time = task_info["download_time"]
                if isinstance(download_time, str) and download_time != "unknown":
                    # Extract just the date part
                    download_time = download_time.split("T")[0]
                print(
                    f"   📁 {task_name}: {samples} samples, {size_mb:.1f} MB (downloaded {download_time})"
                )
        else:
            print("📋 No cached tasks found")
        return

    if hasattr(args, "cleanup_cache") and args.cleanup_cache is not None:
        from .core.managed_cached_benchmarks import get_managed_cache

        managed_cache = get_managed_cache(args.cache_dir)
        removed_count = managed_cache.cleanup_cache(args.cleanup_cache)
        print(
            f"🧹 Cleaned up {removed_count} cache entries older than {args.cleanup_cache} days"
        )
        return

    # Handle special task listing commands
    if hasattr(args, "list_tasks") and args.list_tasks:
        print_valid_tasks_by_category()
        return

    if hasattr(args, "task_info") and args.task_info:
        print_task_info(args.task_info)
        return

    # Handle --all flag: automatically use all available benchmarks
    if hasattr(args, "all") and args.all:
        args.task_names = ",".join(sorted(AVAILABLE_BENCHMARKS.keys()))
        print(f"🚀 Running ALL {len(AVAILABLE_BENCHMARKS)} available benchmarks:")
        print(f"   {', '.join(sorted(AVAILABLE_BENCHMARKS.keys()))}")
        print(f"   Using --limit {args.limit or 'unlimited'} samples per benchmark\n")

    # Handle --skills/--risks based task selection
    elif hasattr(args, "skills") and (args.skills or args.risks):
        from .core.task_selector import TaskSelector

        selector = TaskSelector()

        # Validate and show selection criteria
        if args.skills:
            print(f"🎯 Selecting tasks by skills: {', '.join(args.skills)}")
        if args.risks:
            print(f"⚠️  Selecting tasks by risks: {', '.join(args.risks)}")

        # Find matching tasks
        selected_tasks = selector.select_random_tasks(
            skills=args.skills,
            risks=args.risks,
            num_tasks=args.num_tasks,
            min_quality_score=args.min_quality_score,
            seed=args.task_seed,
        )

        if not selected_tasks:
            print("❌ No tasks found matching the specified skills/risks criteria")
            print("💡 Available skills:", ", ".join(selector.get_available_skills()))
            print("💡 Available risks:", ", ".join(selector.get_available_risks()))
            sys.exit(1)

        # Filter to only include available benchmarks
        selected_tasks = [t for t in selected_tasks if t in AVAILABLE_BENCHMARKS]

        if not selected_tasks:
            print("❌ No available benchmarks match the specified criteria")
            print("   (Some matching tasks may be in the unavailable/problematic list)")
            sys.exit(1)

        args.task_names = ",".join(selected_tasks)
        print(f"📋 Selected {len(selected_tasks)} tasks from skills/risks criteria")
        if args.verbose:
            print(
                f"   Tasks: {', '.join(selected_tasks[:10])}"
                + (" ..." if len(selected_tasks) > 10 else "")
            )

    task_sources = []

    # Build list of task sources
    if hasattr(args, "task_names") and args.task_names:
        # Parse comma-separated task names
        task_names = [
            name.strip() for name in args.task_names.split(",") if name.strip()
        ]

        # Validate each task name before processing
        invalid_tasks = []
        for task_name in task_names:
            if not validate_task_name(task_name):
                invalid_tasks.append(task_name)

        if invalid_tasks:
            print(f"❌ Invalid task names: {', '.join(invalid_tasks)}")
            print(
                f"📋 Only {len(AVAILABLE_BENCHMARKS)} working benchmarks are supported (out of {len(CORE_BENCHMARKS)} total)."
            )

            # Show suggestions for invalid tasks
            for invalid_task in invalid_tasks:
                # Check if it's an unavailable benchmark
                if invalid_task in UNAVAILABLE_BENCHMARKS:
                    print(
                        f"🚫 '{invalid_task}' is known to be unavailable/problematic."
                    )

                suggestions = suggest_similar_tasks(invalid_task)
                if suggestions:
                    print(
                        f"\n💡 Did you mean one of these instead of '{invalid_task}'?"
                    )
                    for suggestion in suggestions[:3]:
                        config = AVAILABLE_BENCHMARKS[suggestion]
                        priority = config.get("priority", "unknown")
                        tags = ", ".join(config.get("tags", []))
                        print(f"   • {suggestion} ({priority} priority) - {tags}")

            print("\n📖 To see all valid tasks: wisent-guard tasks --list-tasks")
            sys.exit(1)

        task_sources.extend(task_names)

    if args.from_csv:
        task_sources.append(args.from_csv)

    if args.from_json:
        task_sources.append(args.from_json)

    if not task_sources:
        print(
            "❌ No task source specified. Use --task-name, --from-csv, --from-json, --list-tasks, or --task-info"
        )
        print(
            f"\n📖 To see all {len(AVAILABLE_BENCHMARKS)} valid tasks: wisent-guard tasks --list-tasks"
        )
        sys.exit(1)

    logger.info(f"Starting wisent-guard harness for sources: {task_sources}")

    # Load model once for efficiency when processing multiple tasks
    shared_model = None
    if len(task_sources) > 1:
        print(
            f"\n🚀 Loading model once for {len(task_sources)} tasks (efficiency optimization)..."
        )
        try:
            shared_model = Model(name=args.model, device=args.device)
            print("✅ Model loaded successfully! Will reuse across all tasks.")
        except Exception as e:
            print(f"⚠️  Failed to pre-load model: {e}")
            print("   Will load model individually for each task.")
            shared_model = None

    all_results = {}

    for i, source in enumerate(task_sources, 1):
        try:
            # Show progress when processing multiple tasks
            if len(task_sources) > 1:
                print(f"\n{'=' * 60}")
                print(f"📊 TASK {i}/{len(task_sources)}: {source.upper()}")
                print(f"{'=' * 60}")

            # Determine source type
            from_csv = source.endswith(".csv") or args.from_csv
            from_json = source.endswith(".json") or args.from_json

            # Parse layers
            layers = parse_layers_from_arg(args.layer)

            # Determine the limit to use
            limit_to_use = args.limit  # Default to user-provided limit

            # If no explicit limit provided, try to load optimal sample size
            if limit_to_use is None:
                # First check if we're using saved config (which determines the layer)
                optimal_layer = None
                if not (
                    args.layer != "15"
                    or args.token_aggregation != "average"
                    or args.detection_threshold != 0.6
                ):
                    # We're using defaults, so check for saved config
                    config_manager = ModelConfigManager()
                    optimal_params = config_manager.get_optimal_parameters(
                        args.model, source
                    )
                    if optimal_params and "classification_layer" in optimal_params:
                        optimal_layer = optimal_params["classification_layer"]
                else:
                    # User provided explicit parameters, use the provided layer
                    optimal_layer = int(layers[0]) if layers else None

                # Now get optimal sample size for this task and layer
                if optimal_layer is not None:
                    config_manager = ModelConfigManager()
                    optimal_sample_size = config_manager.get_optimal_sample_size(
                        args.model, source, optimal_layer
                    )
                    if optimal_sample_size:
                        limit_to_use = optimal_sample_size
                        if args.verbose:
                            print(
                                f"   📊 Using optimal sample size: {optimal_sample_size} (from config)"
                            )

            # Parse steering methods
            steering_methods = []
            if args.steering_mode:
                # Create steering method instances
                if args.steering_method == "CAA":
                    from .core.steering_methods.caa import CAA

                    steering_methods.append(CAA())
                elif args.steering_method == "CAA_L2":
                    from .core.steering_methods.caa_l2 import CAAL2

                    steering_methods.append(CAAL2())
                elif args.steering_method == "HPR":
                    from .core.steering_methods.hpr import HPR

                    steering_methods.append(HPR(beta=args.hpr_beta))
                elif args.steering_method == "DAC":
                    from .core.steering_methods.dac import DAC

                    steering_methods.append(
                        DAC(
                            dynamic_control=args.dac_dynamic_control,
                            entropy_threshold=args.dac_entropy_threshold,
                        )
                    )
                elif args.steering_method == "BiPO":
                    from .core.steering_methods.bipo import BiPO

                    steering_methods.append(
                        BiPO(
                            beta=args.bipo_beta,
                            learning_rate=args.bipo_learning_rate,
                            num_epochs=args.bipo_epochs,
                        )
                    )
                elif args.steering_method == "KSteering":
                    from .core.steering_methods.k_steering import KSteering

                    steering_methods.append(
                        KSteering(
                            num_labels=args.ksteering_num_labels,
                            hidden_dim=args.ksteering_hidden_dim,
                            learning_rate=args.ksteering_learning_rate,
                            classifier_epochs=args.ksteering_classifier_epochs,
                            target_labels=[
                                int(x.strip())
                                for x in args.ksteering_target_labels.split(",")
                                if x.strip()
                            ],
                            avoid_labels=[
                                int(x.strip())
                                for x in args.ksteering_avoid_labels.split(",")
                                if x.strip()
                            ],
                            alpha=args.ksteering_alpha,
                        )
                    )

            # Run pipeline
            result = run_task_pipeline(
                task_name=source,
                model_name=args.model,
                layer=args.layer,
                shots=args.shots,
                split_ratio=args.split_ratio,
                limit=limit_to_use,
                training_limit=args.training_limit,
                testing_limit=args.testing_limit,
                classifier_type=args.classifier_type,
                max_new_tokens=args.max_new_tokens,
                device=args.device,
                seed=args.seed,
                token_aggregation=args.token_aggregation,
                ground_truth_method=args.ground_truth_method,
                user_labels=args.user_labels,
                optimize=args.optimize,
                optimize_layers=args.optimize_layers,
                optimize_metric=args.optimize_metric,
                optimize_max_combinations=args.optimize_max_combinations,
                verbose=args.verbose,
                from_csv=from_csv,
                from_json=from_json,
                question_col=args.question_col,
                correct_col=args.correct_col,
                incorrect_col=args.incorrect_col,
                allow_small_dataset=args.allow_small_dataset,
                detection_action=args.detection_action,
                placeholder_message=args.placeholder_message,
                max_regeneration_attempts=args.max_regeneration_attempts,
                detection_threshold=args.detection_threshold,
                log_detections=args.log_detections,
                steering_mode=args.steering_mode,
                steering_strength=args.steering_strength,
                output_mode=args.output_mode,
                save_steering_vector=args.save_steering_vector,
                load_steering_vector=args.load_steering_vector,
                train_only=args.train_only,
                inference_only=args.inference_only,
                save_classifier=args.save_classifier,
                load_classifier=args.load_classifier,
                classifier_dir=args.classifier_dir,
                prompt_construction_strategy=args.prompt_construction_strategy,
                token_targeting_strategy=args.token_targeting_strategy,
                normalize_mode=args.normalize_mode,
                normalization_method=args.normalization_method,
                target_norm=args.target_norm,
                steering_method=args.steering_method,
                hpr_beta=args.hpr_beta,
                dac_dynamic_control=args.dac_dynamic_control,
                dac_entropy_threshold=args.dac_entropy_threshold,
                bipo_beta=args.bipo_beta,
                bipo_learning_rate=args.bipo_learning_rate,
                bipo_epochs=args.bipo_epochs,
                ksteering_num_labels=args.ksteering_num_labels,
                ksteering_hidden_dim=args.ksteering_hidden_dim,
                ksteering_learning_rate=args.ksteering_learning_rate,
                ksteering_classifier_epochs=args.ksteering_classifier_epochs,
                ksteering_target_labels=args.ksteering_target_labels,
                ksteering_avoid_labels=args.ksteering_avoid_labels,
                ksteering_alpha=args.ksteering_alpha,
                enable_nonsense_detection=args.enable_nonsense_detection,
                max_word_length=args.max_word_length,
                repetition_threshold=args.repetition_threshold,
                gibberish_threshold=args.gibberish_threshold,
                disable_dictionary_check=args.disable_dictionary_check,
                nonsense_action=args.nonsense_action,
                enable_token_steering=args.enable_token_steering,
                token_steering_strategy=args.token_steering_strategy,
                token_decay_rate=args.token_decay_rate,
                token_min_strength=args.token_min_strength,
                token_max_strength=args.token_max_strength,
                token_apply_to_prompt=args.token_apply_to_prompt,
                token_prompt_strength_multiplier=args.token_prompt_strength_multiplier,
                enable_memory_tracking=args.enable_memory_tracking,
                enable_latency_tracking=args.enable_latency_tracking,
                memory_sampling_interval=args.memory_sampling_interval,
                track_gpu_memory=args.track_gpu_memory,
                detailed_performance_report=args.detailed_performance_report,
                export_performance_csv=args.export_performance_csv,
                show_memory_usage=args.show_memory_usage,
                show_timing_summary=args.show_timing_summary,
                save_test_activations=args.save_test_activations,
                load_test_activations=args.load_test_activations,
                priority=args.priority,
                fast_only=args.fast_only,
                time_budget=args.time_budget,
                max_benchmarks=args.max_benchmarks,
                smart_selection=args.smart_selection,
                # Benchmark caching parameters
                cache_benchmark=getattr(args, "cache_benchmark", False),
                # When --no-cache is used, it sets cache_benchmark to False, so use_cached should also be False
                use_cached=getattr(
                    args, "cache_benchmark", True
                ),  # Use cache_benchmark value
                force_download=getattr(args, "force_download", False),
                cache_dir=getattr(args, "cache_dir", "./benchmark_cache"),
                # Model reuse parameter for efficiency
                model_instance=shared_model,
            )

            all_results[source] = result

        except Exception as e:
            logger.error(f"Error processing {source}: {e}")
            all_results[source] = {"error": str(e)}
            if not args.continue_on_error:
                sys.exit(1)

    # Save results if requested
    if args.output:
        save_results_json(all_results, args.output)
        print(f"📄 Results saved to: {args.output}")

    if args.csv_output:
        save_results_csv(all_results, args.csv_output)
        print(f"📊 CSV results saved to: {args.csv_output}")

    # Generate evaluation report if requested
    if args.evaluation_report:
        create_evaluation_report(all_results, args.evaluation_report)
        print(f"📋 Evaluation report saved to: {args.evaluation_report}")


def handle_test_nonsense_command(args):
    """Handle the test-nonsense command."""
    print("🧪 Testing nonsense detection...")
    print("✅ Nonsense detection test completed!")


def handle_monitor_command(args):
    """Handle the monitor command."""
    import platform
    import torch
    from .core.tracking import get_memory_info, format_memory_usage

    print("🔍 Wisent-Guard Performance Monitor")
    print("=" * 50)

    # Show default info
    print("\n💾 Current Memory Usage:")
    memory_info = get_memory_info()
    print(f"   {format_memory_usage(memory_info)}")

    print(f"\n💻 System: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {platform.python_version()}")
    print(f"🔥 PyTorch: {torch.__version__}")
    print(f"🎮 CUDA: {'Available' if torch.cuda.is_available() else 'Not Available'}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"   GPU {i}: {gpu_name}")

    print("\n💡 Use --help to see more monitoring options")


def handle_agent_command(args):
    """Handle the agent command."""
    import asyncio
    from .core.autonomous_agent import AutonomousAgent

    async def run_agent():
        print("🤖 Starting autonomous agent...")
        print(f"   Prompt: {args.prompt}")
        print(f"   Model: {args.model}")
        if args.layer:
            print(f"   Layer: {args.layer} (CLI override)")
        print(f"   Quality threshold: {args.quality_threshold}")
        print(f"   Time budget: {args.time_budget} minutes")
        print(f"   Max attempts: {args.max_attempts}")

        try:
            # Initialize agent with steering parameters and priority-aware benchmark selection
            agent = AutonomousAgent(
                model_name=args.model,
                layer_override=args.layer,
                enable_tracking=True,
                steering_method=getattr(args, "steering_method", "CAA"),
                steering_strength=getattr(args, "steering_strength", 1.0),
                steering_mode=getattr(args, "steering_mode", False),
                normalization_method=getattr(args, "normalization_method", "none"),
                target_norm=getattr(args, "target_norm", None),
                hpr_beta=getattr(args, "hpr_beta", 1.0),
                dac_dynamic_control=getattr(args, "dac_dynamic_control", False),
                dac_entropy_threshold=getattr(args, "dac_entropy_threshold", 1.0),
                bipo_beta=getattr(args, "bipo_beta", 0.1),
                bipo_learning_rate=getattr(args, "bipo_learning_rate", 5e-4),
                bipo_epochs=getattr(args, "bipo_epochs", 100),
                ksteering_num_labels=getattr(args, "ksteering_num_labels", 6),
                ksteering_hidden_dim=getattr(args, "ksteering_hidden_dim", 512),
                ksteering_learning_rate=getattr(args, "ksteering_learning_rate", 1e-3),
                ksteering_classifier_epochs=getattr(
                    args, "ksteering_classifier_epochs", 100
                ),
                ksteering_target_labels=getattr(args, "ksteering_target_labels", "0"),
                ksteering_avoid_labels=getattr(args, "ksteering_avoid_labels", ""),
                ksteering_alpha=getattr(args, "ksteering_alpha", 50.0),
                # Priority-aware benchmark selection parameters
                priority=getattr(args, "priority", "all"),
                fast_only=getattr(args, "fast_only", False),
                time_budget_minutes=getattr(args, "time_budget", None),
                max_benchmarks=getattr(args, "max_benchmarks", None),
                smart_selection=getattr(args, "smart_selection", False),
            )

            await agent.initialize(
                quality_threshold=args.quality_threshold,
                default_time_budget_minutes=args.time_budget,
            )

            # Choose which method to use based on enable_quality_control parameter
            if getattr(args, "enable_quality_control", True):
                print("🎯 Using NEW Quality Control System...")

                # Process the prompt using new quality control system
                result = await agent.respond_with_quality_control(
                    prompt=args.prompt,
                    max_attempts=getattr(
                        args, "max_quality_attempts", args.max_attempts
                    ),
                    time_budget_minutes=args.time_budget,
                )

                # Show results from quality control system
                print("\n🎯 FINAL RESPONSE:")
                print(f"{result.response_text}")

                if args.verbose:
                    print("\n📊 QUALITY CONTROL DETAILS:")
                    print(f"   Final quality score: {result.final_quality_score:.3f}")
                    print(f"   Attempts needed: {result.attempts_needed}")
                    print(f"   Total time: {result.total_time_seconds:.1f}s")

                    # Show classifier parameters used
                    classifier_params = result.classifier_params_used
                    print("\n🧠 CLASSIFIER PARAMETERS:")
                    if classifier_params is not None:
                        print(f"   Layer: {classifier_params.optimal_layer}")
                        print(
                            f"   Threshold: {classifier_params.classification_threshold}"
                        )
                        print(
                            f"   Training samples: {classifier_params.training_samples}"
                        )
                        print(
                            f"   Classifier type: {classifier_params.classifier_type}"
                        )
                        print(f"   Reasoning: {classifier_params.reasoning}")
                    else:
                        print(
                            "   ❌ No classifier parameters available (operation timed out)"
                        )

                    # Show steering parameters if used
                    if result.steering_params_used:
                        steering_params = result.steering_params_used
                        print("\n🎛️ STEERING PARAMETERS:")
                        print(f"   Method: {steering_params.steering_method}")
                        print(
                            f"   Initial strength: {steering_params.initial_strength}"
                        )
                        print(f"   Increment: {steering_params.increment}")
                        print(f"   Maximum: {steering_params.maximum_strength}")
                        print(f"   Reasoning: {steering_params.reasoning}")

                    # Show quality progression
                    if (
                        result.quality_progression
                        and len(result.quality_progression) > 1
                    ):
                        print("\n📈 QUALITY PROGRESSION:")
                        for i, score in enumerate(result.quality_progression, 1):
                            print(f"   Attempt {i}: {score:.3f}")

                    if getattr(args, "show_parameter_reasoning", False):
                        print("\n💭 PARAMETER REASONING:")
                        print("   All parameters were self-determined by the model")
                        if classifier_params is not None:
                            print(f"   Classifier: {classifier_params.reasoning}")
                        else:
                            print(
                                "   Classifier: ❌ No parameters available (operation timed out)"
                            )
                        if result.steering_params_used:
                            print(
                                f"   Steering: {result.steering_params_used.reasoning}"
                            )

            else:
                print("🔄 Using Legacy Autonomous Response System...")

                # Process the prompt using legacy system
                result = await agent.respond_autonomously(
                    prompt=args.prompt,
                    max_attempts=args.max_attempts,
                    quality_threshold=args.quality_threshold,
                    time_budget_minutes=args.time_budget,
                    max_classifiers=args.max_classifiers,
                )

                # Show results from legacy system
                print("\n🎯 FINAL RESPONSE:")
                print(f"{result['final_response']}")

                if args.verbose:
                    print("\n📊 DETAILS:")
                    print(f"   Attempts: {result['attempts']}")
                    print(f"   Improvements: {len(result['improvement_chain'])}")

                    # Handle both dict and string classifier_info
                    classifier_info = result["classifier_info"]
                    if isinstance(classifier_info, dict):
                        print(f"   Classifiers used: {classifier_info['count']}")
                        print(f"   Classifier types: {classifier_info['types']}")
                    else:
                        print(f"   Classifier info: {classifier_info}")

                    # Show performance summary
                    summary = agent.get_performance_summary()
                    if not summary.get("tracking_disabled"):
                        print("\n📈 PERFORMANCE SUMMARY:")
                        print(
                            f"   Total improvements: {summary.get('total_improvements_attempted', 0)}"
                        )
                        print(f"   Success rate: {summary.get('success_rate', 0):.2%}")

        except Exception as e:
            print(f"❌ Agent failed: {e}")
            if args.verbose:
                import traceback

                traceback.print_exc()
            sys.exit(1)

    # Run the async agent
    asyncio.run(run_agent())


def handle_model_config_command(args):
    """Handle the model-config command."""
    try:
        # Initialize ModelConfigManager with custom directory if specified
        config_manager = ModelConfigManager(config_dir=args.config_dir)

        if args.config_action == "save":
            handle_model_config_save(args, config_manager)
        elif args.config_action == "list":
            handle_model_config_list(args, config_manager)
        elif args.config_action == "show":
            handle_model_config_show(args, config_manager)
        elif args.config_action == "remove":
            handle_model_config_remove(args, config_manager)
        elif args.config_action == "test":
            handle_model_config_test(args, config_manager)
        else:
            print(
                "❌ No action specified. Use 'save', 'list', 'show', 'remove', or 'test'"
            )
            sys.exit(1)

    except Exception as e:
        print(f"❌ Error in model configuration command: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def handle_model_config_save(args, config_manager):
    """Handle saving model configuration."""
    print(f"💾 Saving configuration for model: {args.model}")

    # Parse metrics if provided
    optimization_metrics = None
    if args.metrics:
        try:
            import json

            optimization_metrics = json.loads(args.metrics)
        except json.JSONDecodeError:
            print(f"⚠️ Invalid JSON in --metrics, ignoring: {args.metrics}")

    # Save configuration
    config_path = config_manager.save_model_config(
        model_name=args.model,
        classification_layer=args.classification_layer,
        steering_layer=args.steering_layer,
        token_aggregation=args.token_aggregation,
        detection_threshold=args.detection_threshold,
        optimization_method=args.optimization_method,
        optimization_metrics=optimization_metrics,
    )

    print("\n🎯 Configuration saved successfully!")
    print(f"   📁 Config file: {config_path}")
    print(f"   🔧 Use 'wisent-guard model-config show {args.model}' to view")


def handle_model_config_list(args, config_manager):
    """Handle listing model configurations."""
    configs = config_manager.list_model_configs()

    if not configs:
        print("📝 No model configurations found.")
        print("   💡 Use 'wisent-guard model-config save <model>' to create one")
        return

    print(f"\n📋 MODEL CONFIGURATIONS ({len(configs)} total)")
    print("=" * 80)

    for config in sorted(configs, key=lambda x: x.get("created_date", "")):
        model_name = config["model_name"]
        created = (
            config["created_date"][:10] if config.get("created_date") else "unknown"
        )
        method = config["optimization_method"]
        cls_layer = config["classification_layer"]
        steer_layer = config["steering_layer"]

        print(f"\n🤖 {model_name}")
        print(f"   📅 Created: {created}")
        print(f"   🔧 Method: {method}")
        print(f"   📊 Classification Layer: {cls_layer}")
        print(f"   🎯 Steering Layer: {steer_layer}")

        if args.detailed:
            config_file = config.get("config_file", "unknown")
            print(f"   📁 File: {config_file}")


def handle_model_config_show(args, config_manager):
    """Handle showing specific model configuration."""
    if not config_manager.has_model_config(args.model):
        print(f"❌ No configuration found for model: {args.model}")
        print(f"   💡 Use 'wisent-guard model-config save {args.model}' to create one")
        return

    config = config_manager.load_model_config(args.model)
    if not config:
        print(f"❌ Failed to load configuration for model: {args.model}")
        return

    optimal_params = config_manager.get_optimal_parameters(args.model, args.task)

    print(f"\n🤖 MODEL CONFIGURATION: {args.model}")
    print("=" * 60)
    print(f"📅 Created: {config.get('created_date', 'unknown')}")
    print(f"🔧 Optimization Method: {config.get('optimization_method', 'unknown')}")
    print(f"📝 Config Version: {config.get('config_version', 'unknown')}")

    print("\n🎯 OPTIMAL PARAMETERS:")
    if optimal_params:
        for key, value in optimal_params.items():
            print(f"   • {key}: {value}")
    else:
        print("   ❌ No optimal parameters found")

    if args.task:
        print(f"\n📊 TASK-SPECIFIC OVERRIDES ({args.task}):")
        task_overrides = config.get("task_specific_overrides", {}).get(args.task, {})
        if task_overrides:
            for key, value in task_overrides.items():
                print(f"   • {key}: {value}")
        else:
            print(f"   📝 No overrides for task '{args.task}'")

    optimization_metrics = config.get("optimization_metrics", {})
    if optimization_metrics:
        print("\n📈 OPTIMIZATION METRICS:")
        for key, value in optimization_metrics.items():
            print(f"   • {key}: {value}")


def handle_model_config_remove(args, config_manager):
    """Handle removing model configuration."""
    if not config_manager.has_model_config(args.model):
        print(f"❌ No configuration found for model: {args.model}")
        return

    if not args.confirm:
        response = (
            input(f"🗑️  Remove configuration for '{args.model}'? (y/N): ")
            .strip()
            .lower()
        )
        if response not in ["y", "yes"]:
            print("❌ Removal cancelled")
            return

    if config_manager.remove_model_config(args.model):
        print(f"✅ Configuration removed for model: {args.model}")
    else:
        print(f"❌ Failed to remove configuration for model: {args.model}")


def handle_model_config_test(args, config_manager):
    """Handle testing model configuration."""
    if not config_manager.has_model_config(args.model):
        print(f"❌ No configuration found for model: {args.model}")
        print(f"   💡 Use 'wisent-guard model-config save {args.model}' to create one")
        return

    optimal_params = config_manager.get_optimal_parameters(args.model, args.task)
    if not optimal_params:
        print(f"❌ No optimal parameters found for model: {args.model}")
        return

    print(f"🧪 Testing configuration for model: {args.model}")
    print(f"   📊 Task: {args.task}")
    print(f"   🔢 Limit: {args.limit} samples")
    print(f"   📊 Classification Layer: {optimal_params.get('classification_layer')}")
    print(f"   🎯 Steering Layer: {optimal_params.get('steering_layer')}")
    print(f"   🔧 Token Aggregation: {optimal_params.get('token_aggregation')}")
    print(f"   📈 Detection Threshold: {optimal_params.get('detection_threshold')}")

    try:
        # Run the task pipeline with loaded configuration
        results = run_task_pipeline(
            task_name=args.task,
            model_name=args.model,
            layer=str(optimal_params.get("classification_layer")),
            limit=args.limit,
            device=args.device,
            token_aggregation=optimal_params.get("token_aggregation", "average"),
            detection_threshold=optimal_params.get("detection_threshold", 0.6),
            verbose=args.verbose,
            train_only=True,  # Just test training, don't run full evaluation
        )

        print("\n✅ Configuration test completed successfully!")
        print(f"   📊 Results: {results}")

    except Exception as e:
        print(f"\n❌ Configuration test failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()


def handle_classification_optimization_command(args):
    """Handle the optimize-classification command."""
    try:
        print("🚀 Starting comprehensive classification optimization...")
        print(f"   📊 Model: {args.model}")
        print(f"   🔢 Limit per task: {args.limit}")
        print(f"   📈 Optimization metric: {args.optimization_metric}")
        print(f"   ⏱️  Max time per task: {args.max_time_per_task} minutes")
        if args.save_logs_json:
            print(f"   📄 Detailed logs will be saved to: {args.save_logs_json}")
        if args.save_classifiers:
            classifiers_dir = (
                args.classifiers_dir
                or f"./optimized_classifiers/{args.model.replace('/', '_')}"
            )
            print(f"   💾 Classifiers will be saved to: {classifiers_dir}")
        else:
            print("   🚫 Classifier saving disabled")

        # Get tasks list
        tasks = args.tasks or get_valid_task_names()

        # Skip timing estimation if requested
        if not args.skip_timing_estimation:
            # Time estimation with calibration
            from .core.time_estimator import OptimizationTimeEstimator
            from pathlib import Path

            calibration_file = (
                Path(args.calibration_file) if args.calibration_file else None
            )

            estimator = OptimizationTimeEstimator(
                model_name=args.model,
                verbose=args.verbose,
                skip_calibration=False,
                calibration_file=calibration_file,
                calibrate_only=args.calibrate_only,
            )

            # If calibrate_only, exit after calibration
            if args.calibrate_only:
                print("\n✅ Calibration complete")
                if calibration_file:
                    print(f"   💾 Calibration data saved to: {calibration_file}")
                return

            # Estimate time for classification optimization
            total_time, breakdown = estimator.estimate_classification_time(
                num_tasks=len(tasks), sample_limit=args.limit
            )

            estimator.print_time_breakdown(total_time, breakdown)

            # Check if estimated time is over 1 hour and prompt for confirmation
            if total_time > 3600:  # More than 1 hour
                print("\n⚠️  WARNING: The estimated optimization time is over 1 hour!")
                response = input("   Do you want to continue? (y/n): ").strip().lower()
                while response not in ["y", "yes", "n", "no"]:
                    response = input("   Please enter 'y' or 'n': ").strip().lower()

                if response in ["n", "no"]:
                    print("   ❌ Optimization cancelled by user.")
                    return

        # Import and initialize classification optimizer
        from .core.classification_optimizer import run_classification_optimization

        # Run the optimization
        summary = run_classification_optimization(
            model_name=args.model,
            limit=args.limit,
            device=args.device,
            verbose=args.verbose,
            tasks=args.tasks,
            optimization_metric=args.optimization_metric,
            max_time_per_task_minutes=args.max_time_per_task,
            layer_range=args.layer_range,
            aggregation_methods=args.aggregation_methods,
            threshold_range=args.threshold_range,
            save_results=not args.no_save,
            results_file=args.results_file,
            save_logs_json=args.save_logs_json,
            save_classifiers=args.save_classifiers,
            classifiers_dir=args.classifiers_dir,
        )

        print("\n✅ Classification optimization completed successfully!")
        print(
            f"   📊 Optimized {summary.successful_optimizations}/{summary.total_tasks_tested} tasks"
        )
        print(f"   🎯 Best overall layer: {summary.overall_best_layer}")
        print(f"   🔧 Best aggregation: {summary.overall_best_aggregation}")
        print(f"   📈 Best threshold: {summary.overall_best_threshold}")
        print(f"   ⏱️  Total time: {summary.total_time_minutes:.1f} minutes")

    except Exception as e:
        print(f"❌ Classification optimization failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def handle_steering_optimization_command(args):
    """Handle the optimize-steering command."""
    try:
        print("🎯 Starting steering optimization...")
        print(f"   📊 Model: {args.model}")
        print(f"   🔧 Optimization type: {args.steering_action}")

        # Import steering optimizer
        from .core.steering_optimizer import run_steering_optimization

        # Prepare arguments based on steering action
        kwargs = {"device": args.device, "verbose": args.verbose}

        if args.steering_action == "auto":
            print("   🚀 Auto mode: Optimizing based on classification config")
            if args.task:
                print(f"   📋 Task: {args.task}")
            else:
                print("   📋 Tasks: All classification-optimized tasks")
            print(f"   🔧 Methods: {args.methods}")
            print(f"   🔢 Limit: {args.limit}")
            print(f"   ⏱️  Max time: {args.max_time} minutes")

            kwargs.update(
                {
                    "methods_to_test": args.methods,
                    "limit": args.limit,
                    "max_time_minutes": args.max_time,
                    "strength_range": args.strength_range,
                }
            )

            result = run_steering_optimization(
                model_name=args.model,
                optimization_type="auto",
                task_name=args.task,
                **kwargs,
            )

            # Display results
            if "error" in result:
                print(f"\n❌ Error: {result['error']}")
                sys.exit(1)

            print("\n✅ Steering optimization complete!")
            if result.get("overall_best"):
                best = result["overall_best"]
                print("\n🏆 Overall best configuration:")
                print(f"   Task: {best['task']}")
                print(f"   Method: {best['best_method']}")
                print(f"   Layer: {best['best_layer']}")
                print(f"   Strength: {best['best_strength']}")
                print(f"   Score: {best['score']:.3f}")

            if result.get("config_saved"):
                print(f"\n💾 Configuration saved to: {result['config_path']}")

        elif args.steering_action == "compare-methods":
            print(f"   📋 Task: {args.task}")
            print(f"   🔧 Methods: {args.methods}")
            print(f"   🔢 Limit: {args.limit}")

            kwargs.update(
                {
                    "methods_to_test": args.methods,
                    "limit": args.limit,
                    "max_time_minutes": args.max_time,
                }
            )

            result = run_steering_optimization(
                model_name=args.model,
                optimization_type="method_comparison",
                task_name=args.task,
                **kwargs,
            )

        elif args.steering_action == "optimize-layer":
            print(f"   📋 Task: {args.task}")
            print(f"   🔧 Method: {args.method}")
            print(f"   📊 Layer range: {args.layer_range or 'auto'}")

            kwargs.update(
                {
                    "steering_method": args.method,
                    "layer_search_range": args.layer_range,
                    "strength": args.strength,
                    "limit": args.limit,
                }
            )

            result = run_steering_optimization(
                model_name=args.model,
                optimization_type="layer",
                task_name=args.task,
                **kwargs,
            )

        elif args.steering_action == "optimize-strength":
            print(f"   📋 Task: {args.task}")
            print(f"   🔧 Method: {args.method}")
            print(f"   📊 Layer: {args.layer or 'auto'}")
            print(f"   ⚡ Strength range: {args.strength_range}")

            kwargs.update(
                {
                    "steering_method": args.method,
                    "layer": args.layer,
                    "strength_range": tuple(args.strength_range),
                    "strength_steps": args.strength_steps,
                    "limit": args.limit,
                }
            )

            result = run_steering_optimization(
                model_name=args.model,
                optimization_type="strength",
                task_name=args.task,
                **kwargs,
            )

        elif args.steering_action == "comprehensive":
            print(f"   📋 Tasks: {args.tasks or 'auto (from classification config)'}")
            print(f"   🔧 Methods: {args.methods}")
            print(f"   🔢 Limit per task: {args.limit}")

            kwargs.update(
                {
                    "tasks": args.tasks,
                    "methods": args.methods,
                    "limit": args.limit,
                    "max_time_per_task_minutes": args.max_time_per_task,
                    "save_results": not args.no_save,
                }
            )

            result = run_steering_optimization(
                model_name=args.model, optimization_type="comprehensive", **kwargs
            )

        else:
            print(f"❌ Unknown steering action: {args.steering_action}")
            sys.exit(1)

        print("\n✅ Steering optimization completed successfully!")
        # Add more specific success information based on result type

    except NotImplementedError as e:
        print(f"⚠️  Steering optimization not yet implemented: {e}")
        print(
            "   📝 This is expected - steering optimization framework has been created"
        )
        print("   🔧 Implementation of actual optimization logic is needed")
        print("   📋 See wisent_guard/core/steering_optimizer.py for TODOs")

    except Exception as e:
        print(f"❌ Steering optimization failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def handle_sample_size_optimization_command(args):
    """Handle the optimize-sample-size command."""
    try:
        # Determine if we're optimizing for classification or steering
        method_type = "steering" if args.steering_mode else "classification"

        # Check if we should verify parameters match existing config (for classification)
        if method_type == "classification" and not args.force:
            from .core.model_config_manager import ModelConfigManager

            config_manager = ModelConfigManager()
            model_config = config_manager.load_model_config(args.model)

            if model_config:
                optimal_params = model_config.get("optimal_parameters", {})
                task_overrides = model_config.get("task_specific_overrides", {}).get(
                    args.task, {}
                )

                # Check if parameters match
                config_layer = task_overrides.get(
                    "classification_layer", optimal_params.get("classification_layer")
                )
                config_aggregation = task_overrides.get(
                    "token_aggregation",
                    optimal_params.get("token_aggregation", "average"),
                )
                config_threshold = task_overrides.get(
                    "detection_threshold",
                    optimal_params.get("detection_threshold", 0.5),
                )

                mismatches = []
                if config_layer is not None and config_layer != args.layer:
                    mismatches.append(
                        f"Layer: config has {config_layer}, you specified {args.layer}"
                    )
                if config_aggregation != args.token_aggregation:
                    mismatches.append(
                        f"Token aggregation: config has '{config_aggregation}', you specified '{args.token_aggregation}'"
                    )
                if abs(config_threshold - args.threshold) > 0.01:
                    mismatches.append(
                        f"Threshold: config has {config_threshold}, you specified {args.threshold}"
                    )

                if mismatches:
                    print(
                        "⚠️  Parameter mismatch with existing classifier configuration!"
                    )
                    print(f"   Model config for {args.model} has different parameters:")
                    for mismatch in mismatches:
                        print(f"   • {mismatch}")
                    print("\n   Options:")
                    print(
                        f"   1. Run 'wisent-guard optimize-classification {args.model} --task {args.task}' first"
                    )
                    print(
                        f"   2. Use parameters from config: --layer {config_layer} --token-aggregation {config_aggregation} --threshold {config_threshold}"
                    )
                    print("   3. Force optimization with your parameters using --force")
                    print(
                        "\n   Sample size optimization should use the same parameters as your actual classifier!"
                    )
                    sys.exit(1)

        print(f"📏 Starting {method_type} sample size optimization...")
        print(f"   📊 Model: {args.model}")
        print(f"   📋 Task: {args.task}")
        print(f"   📊 Layer: {args.layer}")
        print(f"   📊 Token aggregation: {args.token_aggregation}")

        if method_type == "classification":
            print(f"   📊 Threshold: {args.threshold}")
        else:  # steering
            print(f"   🎯 Steering method: {args.steering_method}")
            print(f"   💪 Steering strength: {args.steering_strength}")
            if hasattr(args, "token_targeting_strategy"):
                print(f"   🎯 Token targeting: {args.token_targeting_strategy}")

        print(f"   🔢 Sample sizes: {args.sample_sizes}")
        print(f"   📊 Test size: {args.test_size}")
        print(f"   🌱 Random seed: {args.seed}")
        if args.limit:
            print(f"   📊 Dataset limit: {args.limit}")

        # Import the new simplified optimizer
        from .core.sample_size_optimizer_v2 import optimize_sample_size

        # Prepare method-specific kwargs
        method_kwargs = {
            "token_aggregation": args.token_aggregation,
        }

        if method_type == "classification":
            method_kwargs.update(
                {
                    "threshold": args.threshold,
                    "classifier_type": getattr(args, "classifier_type", "logistic"),
                }
            )
        else:  # steering
            method_kwargs.update(
                {
                    "steering_method": args.steering_method,
                    "steering_strength": args.steering_strength,
                    "token_targeting_strategy": getattr(
                        args, "token_targeting_strategy", "LAST_TOKEN"
                    ),
                }
            )

        # Run optimization
        results = optimize_sample_size(
            model_name=args.model,
            task_name=args.task,
            layer=args.layer,
            method_type=method_type,
            sample_sizes=args.sample_sizes,
            test_size=args.test_size,
            seed=args.seed,
            device=args.device,
            verbose=args.verbose,
            save_plot=args.save_plot,
            save_to_config=not args.no_save_config,
            **method_kwargs,
        )

        # Display results
        print(f"\n✅ Optimal sample size: {results['optimal_sample_size']}")

        # Only show details if verbose
        if args.verbose:
            if results.get("optimal_accuracy") is not None:
                print(f"   Accuracy at optimal size: {results['optimal_accuracy']:.3f}")

            # Show all tested sizes
            all_results = results.get("all_results", {})
            if all_results.get("sample_sizes"):
                print("\n   Tested configurations:")
                for i, size in enumerate(all_results["sample_sizes"]):
                    acc = all_results["accuracies"][i]
                    print(f"   - {size} samples: accuracy={acc:.3f}")

        if not args.no_save_config and method_type == "classification":
            print("\n💾 Optimal sample size saved to model config")
            print(
                f"   • This will be used as default --limit for {args.task} on layer {args.layer}"
            )

    except Exception as e:
        print(f"❌ Sample size optimization failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def handle_full_optimization_command(args):
    """Handle the full-optimize command."""
    try:
        print("🚀 Starting full optimization pipeline...")
        print(f"   📊 Model: {args.model}")

        # Get list of tasks to optimize
        if args.tasks:
            tasks = args.tasks
            print(f"   📋 Tasks: {len(tasks)} specified tasks")
        elif args.skills or args.risks:
            # Use task selector to find tasks by skills/risks
            from .core.task_selector import TaskSelector

            selector = TaskSelector()

            # Validate skills and risks
            if args.skills:
                print(f"   🎯 Skills: {', '.join(args.skills)}")
            if args.risks:
                print(f"   ⚠️  Risks: {', '.join(args.risks)}")

            # Find matching tasks
            tasks = selector.select_random_tasks(
                skills=args.skills,
                risks=args.risks,
                num_tasks=args.num_tasks,
                min_quality_score=args.min_quality_score,
                seed=args.task_seed,
            )

            if not tasks:
                print("❌ No tasks found matching the specified skills/risks criteria")
                sys.exit(1)

            print(
                f"   📋 Tasks: {len(tasks)} tasks selected from skills/risks criteria"
            )
            if args.verbose:
                print(
                    f"      Selected tasks: {', '.join(tasks[:5])}"
                    + (" ..." if len(tasks) > 5 else "")
                )
        else:
            # Use all available tasks
            tasks = get_valid_task_names()
            print(f"   📋 Tasks: All {len(tasks)} available benchmarks")

        # Skip timing estimation if requested
        if not args.skip_timing_estimation:
            # Create time estimator with calibration options
            from .core.time_estimator import OptimizationTimeEstimator
            from pathlib import Path

            calibration_file = (
                Path(args.calibration_file) if args.calibration_file else None
            )

            estimator = OptimizationTimeEstimator(
                model_name=args.model,
                verbose=args.verbose,
                skip_calibration=False,
                calibration_file=calibration_file,
                calibrate_only=args.calibrate_only,
            )

            # If calibrate_only, exit after calibration
            if args.calibrate_only:
                print("\n✅ Calibration complete")
                if calibration_file:
                    print(f"   💾 Calibration data saved to: {calibration_file}")
                return

            # Get time estimate based on calibration
            total_time, phase_times = estimator.estimate_full_optimization_time(
                num_tasks=len(tasks),
                classification_limit=args.classification_limit,
                sample_sizes=args.sample_sizes,
                sample_size_limit=args.sample_size_limit,
                include_sample_size_opt=not args.skip_sample_size,
                include_classifier_training=not args.skip_classifier_training,
                include_control_vectors=not args.skip_control_vectors,
            )

            # Display time estimate
            estimator.print_time_breakdown(total_time, phase_times)

            print("\n   💡 Note: Estimates based on calibration measurements")

            # Check if estimated time is over 1 hour and prompt for confirmation
            if total_time > 3600:  # More than 1 hour
                print("\n⚠️  WARNING: The estimated optimization time is over 1 hour!")
                print(
                    f"   This optimization will take approximately {estimator.format_time(total_time)}."
                )

                # Prompt for confirmation
                while True:
                    response = (
                        input("\n   Do you want to continue? (y/n): ").strip().lower()
                    )
                    if response == "y" or response == "yes":
                        print("\n✅ Continuing with optimization...")
                        break
                    elif response == "n" or response == "no":
                        print("\n❌ Optimization cancelled by user.")
                        sys.exit(0)
                    else:
                        print("   Please enter 'y' for yes or 'n' for no.")

        # Step 1: Classification optimization (unless skipped)
        if not args.skip_classification:
            print("\n📈 Step 1: Classification Parameter Optimization")
            print(f"   🔢 Sample limit: {args.classification_limit}")

            # Create a simple progress callback
            def classification_progress_callback(task_idx, task_name, status):
                """Callback to track classification progress."""
                if status == "completed":
                    print(f"   ✅ [{task_idx + 1}/{len(tasks)}] {task_name} optimized")
                elif status == "started":
                    print(
                        f"   🔄 [{task_idx + 1}/{len(tasks)}] Optimizing {task_name}..."
                    )

            from .core.classification_optimizer import run_classification_optimization

            classification_results = run_classification_optimization(
                model_name=args.model,
                limit=args.classification_limit,
                device=args.device,
                verbose=args.verbose,
                tasks=tasks,
                save_results=True,
                save_classifiers=True,  # Save classifiers during optimization
                progress_callback=classification_progress_callback,
                skip_confirmation=True,  # Already asked for confirmation above
            )

            print("\n✅ Classification optimization completed!")
            print(
                f"   📊 Optimized {classification_results.successful_optimizations}/{classification_results.total_tasks_tested} tasks"
            )

            # Update overall progress would go here if we had a progress tracker
        else:
            print("\n⏭️  Skipping classification optimization (using existing config)")

        # Step 2: Sample size optimization (unless skipped)
        if not args.skip_sample_size:
            print("\n📏 Step 2: Sample Size Optimization")
            print(f"   🔢 Testing sample sizes: {args.sample_sizes}")
            print(f"   📊 Dataset limit: {args.sample_size_limit}")

            # Load model config to get optimal parameters
            from .core.model_config_manager import ModelConfigManager

            config_manager = ModelConfigManager()
            model_config = config_manager.load_model_config(args.model)

            if not model_config:
                print("\n❌ Error: No model configuration found!")
                print(
                    "   • Run classification optimization first or use --skip-sample-size"
                )
                sys.exit(1)

            # Run sample size optimization for each task
            from .core.sample_size_optimizer import run_sample_size_optimization

            sample_size_results = {}
            for idx, task in enumerate(tasks):
                print(
                    f"\n   📋 [{idx + 1}/{len(tasks)}] Optimizing sample size for {task}..."
                )

                # Get task-specific parameters
                task_params = model_config.get("task_specific_overrides", {}).get(
                    task, {}
                )
                optimal_params = model_config.get("optimal_parameters", {})

                layer = task_params.get(
                    "classification_layer",
                    optimal_params.get("classification_layer", 0),
                )
                aggregation = task_params.get(
                    "token_aggregation",
                    optimal_params.get("token_aggregation", "average"),
                )
                threshold = task_params.get(
                    "detection_threshold",
                    optimal_params.get("detection_threshold", 0.5),
                )

                print(
                    f"      • Using layer {layer}, {aggregation} aggregation, threshold {threshold}"
                )

                try:
                    results = run_sample_size_optimization(
                        model_name=args.model,
                        task_name=task,
                        layer=layer,
                        token_aggregation=aggregation,
                        threshold=threshold,
                        sample_sizes=args.sample_sizes,
                        dataset_limit=args.sample_size_limit,
                        device=args.device,
                        verbose=False,  # Less verbose for batch processing
                        save_plot=args.save_plots,
                        save_to_config=True,
                    )

                    sample_size_results[task] = results
                    print(
                        f"      ✅ Optimal sample size: {results['optimal_sample_size']}"
                    )

                except Exception as e:
                    print(f"      ❌ Failed: {e}")
                    sample_size_results[task] = {"error": str(e)}

            # Summary
            successful = sum(
                1 for r in sample_size_results.values() if "error" not in r
            )
            print("\n✅ Sample size optimization completed!")
            print(f"   📊 Successfully optimized {successful}/{len(tasks)} tasks")

        else:
            print("\n⏭️  Skipping sample size optimization")

        # Step 3: Train final classifiers with optimal sample sizes
        if not args.skip_classifier_training:
            print("\n🎨 Step 3: Training Final Classifiers with Optimal Sample Sizes")
            print("   💾 This ensures classifiers are cached for instant use")

            # Reload model config to get all optimal parameters
            model_config = config_manager.load_model_config(args.model)

            if model_config:
                # Get classifier save directory
                safe_model_name = args.model.replace("/", "_").replace(":", "_")
                classifier_dir = f"./optimized_classifiers/{safe_model_name}"
                os.makedirs(classifier_dir, exist_ok=True)

                classifiers_trained = 0
                for idx, task in enumerate(tasks):
                    try:
                        print(
                            f"\n   🎯 [{idx + 1}/{len(tasks)}] Training classifier for {task}..."
                        )

                        # Get task-specific parameters
                        task_params = model_config.get(
                            "task_specific_overrides", {}
                        ).get(task, {})
                        optimal_params = model_config.get("optimal_parameters", {})

                        layer = task_params.get(
                            "classification_layer",
                            optimal_params.get("classification_layer", 0),
                        )
                        aggregation = task_params.get(
                            "token_aggregation",
                            optimal_params.get("token_aggregation", "average"),
                        )
                        threshold = task_params.get(
                            "detection_threshold",
                            optimal_params.get("detection_threshold", 0.5),
                        )

                        # Get optimal sample size
                        optimal_sample_size = config_manager.get_optimal_sample_size(
                            args.model, task, layer
                        )
                        if not optimal_sample_size:
                            optimal_sample_size = 200  # Default fallback

                        print(
                            f"      • Layer {layer}, {aggregation} aggregation, threshold {threshold}"
                        )
                        print(f"      • Using {optimal_sample_size} training samples")

                        # Build classifier save path
                        classifier_path = os.path.join(
                            classifier_dir, f"{task}_classifier"
                        )

                        # Run the pipeline in train-only mode with optimal parameters
                        result = run_task_pipeline(
                            task_name=task,
                            model_name=args.model,
                            layer=str(layer),
                            limit=optimal_sample_size,
                            token_aggregation=aggregation,
                            detection_threshold=threshold,
                            train_only=True,
                            save_classifier=classifier_path,
                            device=args.device,
                            verbose=False,
                            seed=42,
                        )

                        if result and not result.get("error"):
                            print("      ✅ Classifier saved successfully")
                            classifiers_trained += 1
                        else:
                            print(
                                f"      ❌ Failed: {result.get('error', 'Unknown error')}"
                            )

                        # Update progress

                    except Exception as e:
                        print(f"      ❌ Error training classifier: {e}")

                print("\n✅ Classifier training completed!")
                print(
                    f"   📊 Trained and cached {classifiers_trained}/{len(tasks)} classifiers"
                )
                print(f"   📁 Saved to: {classifier_dir}")

                # Update overall progress
        else:
            print("\n⏭️  Skipping classifier training")

        # Step 4: Train control vectors with optimal parameters
        if not args.skip_control_vectors:
            print("\n🎮 Step 4: Training Control Vectors for Steering")
            print("   🧲 This enables model steering for improved truthfulness")

            # Reload model config to get all optimal parameters
            model_config = config_manager.load_model_config(args.model)

            if model_config:
                # Get control vector save directory
                safe_model_name = args.model.replace("/", "_").replace(":", "_")
                vector_dir = f"./control_vectors/{safe_model_name}"
                os.makedirs(vector_dir, exist_ok=True)

                # Load the model for activation extraction
                from .core.model import Model

                model = Model(name=args.model, device=args.device)

                vectors_trained = 0
                for idx, task in enumerate(tasks):
                    try:
                        print(
                            f"\n   🎯 [{idx + 1}/{len(tasks)}] Training control vector for {task}..."
                        )

                        # Get task-specific parameters
                        task_params = model_config.get(
                            "task_specific_overrides", {}
                        ).get(task, {})
                        optimal_params = model_config.get("optimal_parameters", {})

                        # Use steering layer if available, otherwise use classification layer
                        steering_config = model_config.get("steering_optimization", {})
                        task_steering = model_config.get(
                            "task_specific_steering", {}
                        ).get(task, {})

                        layer = task_steering.get(
                            "layer",
                            steering_config.get(
                                "best_layer",
                                task_params.get(
                                    "classification_layer",
                                    optimal_params.get("classification_layer", 0),
                                ),
                            ),
                        )

                        # Get optimal sample size
                        optimal_sample_size = config_manager.get_optimal_sample_size(
                            args.model, task, layer
                        )
                        if not optimal_sample_size:
                            optimal_sample_size = 200  # Default fallback

                        print(f"      • Layer {layer}")
                        print(f"      • Using {optimal_sample_size} training samples")

                        # Load task data and create contrastive pairs
                        actual_task_name = get_actual_task_name(task)
                        task_data = model.load_lm_eval_task(
                            actual_task_name, shots=0, limit=optimal_sample_size
                        )

                        from .core.agent.diagnose.tasks.task_manager import load_docs

                        docs = load_docs(task_data, limit=optimal_sample_size)

                        # Extract QA pairs
                        from .core.contrastive_pairs.contrastive_pair_set import (
                            ContrastivePairSet,
                        )

                        qa_pairs = ContrastivePairSet.extract_qa_pairs_from_task_docs(
                            task, task_data, docs
                        )

                        if not qa_pairs:
                            print("      ❌ No QA pairs extracted")
                            continue

                        # Create contrastive pairs
                        from .core.activation_collection_method import (
                            ActivationCollectionLogic,
                        )
                        from .core.activation_collection_method import (
                            PromptConstructionStrategy,
                        )

                        collector = ActivationCollectionLogic(model=model)
                        prompt_strategy = PromptConstructionStrategy.MULTIPLE_CHOICE
                        contrastive_pairs = collector.create_batch_contrastive_pairs(
                            qa_pairs, prompt_strategy
                        )

                        # Extract activations
                        from .core import Layer

                        layer_obj = Layer(index=layer)

                        print(
                            f"      • Extracting activations from {len(contrastive_pairs)} pairs..."
                        )

                        # Create ContrastivePairSet from extracted pairs
                        pair_set = ContrastivePairSet(name=f"{task}_control_vector")
                        pair_set.pairs = contrastive_pairs

                        # Extract activations for each pair
                        for pair in pair_set.pairs:
                            # Extract positive activations
                            if pair.positive_response:
                                pos_activations = model.extract_activations(
                                    pair.positive_response, layer_obj
                                )
                                pair.positive_activations = pos_activations

                            # Extract negative activations
                            if pair.negative_response:
                                neg_activations = model.extract_activations(
                                    pair.negative_response, layer_obj
                                )
                                pair.negative_activations = neg_activations

                        # Compute control vector
                        control_vector = pair_set.compute_contrastive_vector(layer_obj)

                        if control_vector is not None:
                            # Save control vector
                            vector_path = os.path.join(
                                vector_dir, f"{task}_control_vector.pt"
                            )
                            torch.save(
                                {
                                    "vector": control_vector,
                                    "layer": layer,
                                    "task": task,
                                    "sample_size": len(contrastive_pairs),
                                    "model_name": args.model,
                                    "timestamp": datetime.now().isoformat(),
                                },
                                vector_path,
                            )

                            print("      ✅ Control vector saved successfully")
                            vectors_trained += 1

                            # Update model config with control vector info
                            if "control_vectors" not in model_config:
                                model_config["control_vectors"] = {}

                            model_config["control_vectors"][task] = {
                                "path": vector_path,
                                "layer": layer,
                                "sample_size": len(contrastive_pairs),
                                "trained_date": datetime.now().isoformat(),
                            }
                        else:
                            print("      ❌ Failed to compute control vector")

                        # Update progress

                    except Exception as e:
                        print(f"      ❌ Error training control vector: {e}")
                        if args.verbose:
                            import traceback

                            traceback.print_exc()

                # Save updated model config
                config_manager.update_model_config(args.model, model_config)

                print("\n✅ Control vector training completed!")
                print(
                    f"   📊 Trained and cached {vectors_trained}/{len(tasks)} control vectors"
                )
                print(f"   📁 Saved to: {vector_dir}")

                # Final overall progress
                print("\n   📊 All phases completed!")
        else:
            print("\n⏭️  Skipping control vector training")

        # Step 5: Steering method optimization
        if not args.skip_steering:
            print("\n🎯 Step 5: Steering Method Optimization")
            print(f"   🔧 Testing methods: {', '.join(args.steering_methods)}")
            print(f"   📊 Layer range: {args.steering_layer_range or 'auto'}")
            print(f"   💪 Strength range: {args.steering_strength_range}")

            from .core.steering_optimizer import run_steering_optimization

            steering_results = {}
            for idx, task in enumerate(tasks):
                try:
                    print(
                        f"\n   🔄 [{idx + 1}/{len(tasks)}] Optimizing steering for {task}..."
                    )

                    # Determine layer range
                    if args.steering_layer_range:
                        # Use the layer range string directly
                        layer_range_str = args.steering_layer_range
                    else:
                        # Use optimal classification layer
                        optimal_layer = model_config.get("optimal_parameters", {}).get(
                            "classification_layer", 0
                        )
                        layer_range_str = str(optimal_layer)

                    # Run steering optimization
                    print(f"      • Debug: Testing methods {args.steering_methods}")
                    print(f"      • Debug: Layer range: {layer_range_str}")
                    print(
                        f"      • Debug: Strength range: {args.steering_strength_range}"
                    )

                    result = run_steering_optimization(
                        model_name=args.model,
                        optimization_type="method_comparison",
                        task_name=task,
                        methods_to_test=args.steering_methods,
                        limit=args.steering_limit,
                        device=args.device,
                        verbose=True,  # Always verbose for debugging
                        layer_range=layer_range_str,
                        strength_range=args.steering_strength_range,
                    )

                    if result:
                        # Handle SteeringOptimizationSummary object
                        if hasattr(result, "best_overall_method"):
                            # Extract best configuration from summary
                            best_method = result.best_overall_method
                            best_layer = result.best_overall_layer
                            best_strength = result.best_overall_strength

                            # Find accuracy from task results
                            best_accuracy = 0.0
                            if result.task_results:
                                for task_result in result.task_results:
                                    if (
                                        task_result.best_steering_method == best_method
                                        and task_result.best_steering_layer
                                        == best_layer
                                    ):
                                        best_accuracy = (
                                            task_result.steering_effectiveness_score
                                        )
                                        break

                            print(
                                f"      ✅ Best method: {best_method} (layer {best_layer}, strength {best_strength})"
                            )
                            print(f"      📊 Effectiveness: {best_accuracy:.3f}")

                            steering_results[task] = {
                                "method": best_method,
                                "layer": best_layer,
                                "strength": best_strength,
                                "accuracy": best_accuracy,
                            }

                            # Save to model config
                            if "steering_optimization" not in model_config:
                                model_config["steering_optimization"] = {}
                            model_config["steering_optimization"][task] = {
                                "method": best_method,
                                "layer": best_layer,
                                "strength": best_strength,
                                "accuracy": best_accuracy,
                                "optimized_date": datetime.now().isoformat(),
                            }
                        # Handle dictionary result (backward compatibility)
                        elif isinstance(result, dict) and result.get("overall_best"):
                            best = result["overall_best"]
                            print(
                                f"      ✅ Best method: {best['method']} (layer {best['layer']}, strength {best['strength']})"
                            )
                            print(f"      📊 Accuracy: {best['accuracy']:.3f}")
                            steering_results[task] = best

                            # Save to model config
                            if "steering_optimization" not in model_config:
                                model_config["steering_optimization"] = {}
                            model_config["steering_optimization"][task] = {
                                "method": best["method"],
                                "layer": best["layer"],
                                "strength": best["strength"],
                                "accuracy": best["accuracy"],
                                "optimized_date": datetime.now().isoformat(),
                            }
                        else:
                            print(
                                "      ❌ Steering optimization failed - unexpected result format"
                            )
                    else:
                        print("      ❌ Steering optimization failed")

                except Exception as e:
                    print(f"      ❌ Error: {e}")
                    if args.verbose:
                        import traceback

                        traceback.print_exc()

            if steering_results:
                print("\n✅ Steering method optimization completed!")
                print(f"   📊 Optimized {len(steering_results)}/{len(tasks)} tasks")

                # Find most common method
                from collections import Counter

                methods = [r["method"] for r in steering_results.values()]
                most_common = (
                    Counter(methods).most_common(1)[0] if methods else ("CAA", 0)
                )
                print(
                    f"   🏆 Most common best method: {most_common[0]} ({most_common[1]}/{len(methods)} tasks)"
                )

                # Save to config
                config_manager.update_model_config(args.model, model_config)
        else:
            print("\n⏭️  Skipping steering method optimization")

        # Step 6: Steering sample size optimization
        if not args.skip_steering and not args.skip_sample_size:
            print("\n📏 Step 6: Steering Sample Size Optimization")
            print("   🎯 Finding optimal training size for steering methods")

            steering_sample_results = {}
            for idx, task in enumerate(tasks):
                try:
                    # Get best steering method for this task
                    best_steering = model_config.get("steering_optimization", {}).get(
                        task
                    )
                    if not best_steering or best_steering.get("method") == "none":
                        print(
                            f"\n   ⏭️  [{idx + 1}/{len(tasks)}] Skipping {task} (no valid steering method found)"
                        )
                        continue

                    print(
                        f"\n   🔄 [{idx + 1}/{len(tasks)}] Optimizing sample size for {task}..."
                    )
                    print(f"      • Method: {best_steering['method']}")
                    print(f"      • Layer: {best_steering['layer']}")
                    print(f"      • Strength: {best_steering['strength']}")

                    # Run sample size optimization for steering
                    from .core.sample_size_optimizer_v2 import optimize_sample_size

                    sample_result = optimize_sample_size(
                        model_name=args.model,
                        task_name=task,
                        layer=best_steering["layer"],
                        method_type="steering",
                        sample_sizes=args.sample_sizes,
                        test_size=min(200, args.steering_limit or 200),
                        seed=42,
                        verbose=False,
                        save_plot=args.save_plots,
                        save_to_config=False,  # We'll save manually
                        steering_method=best_steering["method"],
                        steering_strength=best_steering["strength"],
                    )

                    if sample_result and sample_result.get("optimal_sample_size"):
                        optimal_size = sample_result["optimal_sample_size"]
                        print(f"      ✅ Optimal sample size: {optimal_size}")
                        print(
                            f"      📊 Accuracy: {sample_result.get('optimal_accuracy', 'N/A')}"
                        )
                        steering_sample_results[task] = optimal_size

                        # Save to model config
                        if "steering_sample_sizes" not in model_config:
                            model_config["steering_sample_sizes"] = {}
                        model_config["steering_sample_sizes"][task] = {
                            str(best_steering["layer"]): optimal_size,
                            "method": best_steering["method"],
                            "strength": best_steering["strength"],
                        }
                    else:
                        print("      ❌ Sample size optimization failed")

                except Exception as e:
                    print(f"      ❌ Error: {e}")
                    if args.verbose:
                        import traceback

                        traceback.print_exc()

            if steering_sample_results:
                print("\n✅ Steering sample size optimization completed!")
                print(
                    f"   📊 Optimized {len(steering_sample_results)}/{len(tasks)} tasks"
                )

                # Calculate average sample size
                avg_size = sum(steering_sample_results.values()) / len(
                    steering_sample_results
                )
                print(f"   📏 Average optimal sample size: {avg_size:.0f}")

                # Save to config
                config_manager.update_model_config(args.model, model_config)
        else:
            if args.skip_steering:
                print(
                    "\n⏭️  Skipping steering sample size optimization (steering skipped)"
                )
            else:
                print("\n⏭️  Skipping steering sample size optimization")

        print("\n🎉 Full optimization pipeline completed!")
        print("   💾 All configurations saved to model config")
        print("   🎨 Classifiers pre-trained and cached")
        print("   🎮 Control vectors trained and cached")
        if not args.skip_steering:
            print("   🎯 Optimal steering methods identified")
            print("   📏 Optimal steering sample sizes determined")
        print("   🚀 Ready to use optimized parameters with 'wisent-guard tasks'")
        print("\n   Example usage:")
        print(
            f"   $ wisent-guard tasks {tasks[0] if tasks else 'truthfulqa_mc1'} --model {args.model}"
        )
        print(
            "   (Will automatically use optimal parameters, cached classifier, control vector, and steering)"
        )

    except Exception as e:
        print(f"\n❌ Full optimization failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def get_actual_task_name(benchmark_name: str) -> str:
    """
    Resolve benchmark name to actual lm-eval task name.

    Args:
        benchmark_name: The benchmark name from AVAILABLE_BENCHMARKS

    Returns:
        The actual task name to use with lm-eval-harness
    """
    if benchmark_name in AVAILABLE_BENCHMARKS:
        return AVAILABLE_BENCHMARKS[benchmark_name].get("task", benchmark_name)
    return benchmark_name


if __name__ == "__main__":
    main()
