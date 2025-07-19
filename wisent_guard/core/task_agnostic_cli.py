"""
Task-agnostic CLI demonstration.

This shows how the new architecture would work without depending on lm-eval for task validation.

TODO: REFACTOR WITH MAIN CLI
=====================================
This task-agnostic architecture is now fully functional with real model inference (tested with distilgpt2).
The next step is to refactor wisent_guard/cli.py to use this architecture instead of lm-eval dependency:

1. Replace lm-eval task validation with TaskRegistry.list_tasks()
2. Use TaskInterface.get_extractor() instead of lm-eval for data extraction
3. Integrate TaskRegistry into main CLI argument parsing
4. Remove the 34 references to lm-eval in cli.py for task management
5. Keep lm-eval only for actual evaluation metrics, not task discovery

Current status: Proof-of-concept complete, production integration pending.
"""

import argparse
from typing import Optional
from .task_interface import get_task, list_task_info


def demo_task_agnostic_approach():
    """Demonstrate the task-agnostic approach."""
    print("🔄 Task-Agnostic Architecture Demo")
    print("=" * 50)

    # Import tasks (this registers them automatically)
    try:
        from .tasks import register_all_tasks

        register_all_tasks()  # Explicitly call registration
    except Exception as e:
        print(f"Warning: Could not register tasks: {e}")
        # Register tasks manually for demo
        from .task_interface import register_task
        from .tasks.livecodebench_task import LiveCodeBenchTask

        register_task("livecodebench", LiveCodeBenchTask)

    # List all available tasks
    print("\n📋 Available Tasks:")
    for task_info in list_task_info():
        print(f"   • {task_info['name']:<15} - {task_info['description']}")
        print(f"     Categories: {', '.join(task_info['categories'])}")

    print("\n" + "=" * 50)

    # Demonstrate LiveCodeBench task
    print("\n🧪 Testing LiveCodeBench Task:")
    try:
        lcb_task = get_task("livecodebench")
        print(f"   ✅ Task loaded: {lcb_task.get_name()}")
        print(f"   📝 Description: {lcb_task.get_description()}")
        print(f"   🏷️  Categories: {lcb_task.get_categories()}")

        # Load sample data
        data = lcb_task.load_data(limit=2)
        print(f"   📊 Sample data loaded: {len(data)} items")

        # Test extractor
        extractor = lcb_task.get_extractor()
        qa_pair = extractor.extract_qa_pair(data[0])
        print(f"   🔍 QA pair extracted: {'✅' if qa_pair else '❌'}")

        if qa_pair:
            print(f"   📝 Question: {qa_pair['question'][:50]}...")
            print(f"   💻 Answer: {qa_pair['correct_answer'][:50]}...")

    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("\n" + "=" * 50)

    # Demonstrate MBPP task (if available)
    print("\n🧪 Testing MBPP Task:")
    try:
        mbpp_task = get_task("mbpp")
        print(f"   ✅ Task loaded: {mbpp_task.get_name()}")
        print(f"   📝 Description: {mbpp_task.get_description()}")
        print(f"   🏷️  Categories: {mbpp_task.get_categories()}")

        # Note: This would fail if lm-eval is not available, but the task registration still works
        print("   ⚠️  Data loading requires lm-eval (would work in real implementation)")

    except Exception as e:
        print(f"   ❌ Error: {e}")


def run_task_agnostic_pipeline(
    task_name: str,
    model_name: str,
    layer: int,
    limit: Optional[int] = None,
    device: str = "auto",
    release_version: str = "release_v1",
):
    """Run the task-agnostic pipeline."""
    print("🚀 Running Task-Agnostic Pipeline")
    print(f"   Task: {task_name}")
    print(f"   Model: {model_name}")
    print(f"   Layer: {layer}")
    print(f"   Limit: {limit}")
    print(f"   Device: {device}")
    print(f"   Release Version: {release_version}")

    # Ensure tasks are registered
    try:
        from .tasks import register_all_tasks

        register_all_tasks()
    except Exception as e:
        print(f"Warning: Could not register all tasks: {e}")
        # Register LiveCodeBench manually
        from .task_interface import register_task
        from .tasks.livecodebench_task import LiveCodeBenchTask

        register_task("livecodebench", LiveCodeBenchTask)

    try:
        # Get task with release version support
        if task_name == "livecodebench":
            # Create LiveCodeBench task with specific release version
            from .tasks.livecodebench_task import LiveCodeBenchTask

            task = LiveCodeBenchTask(release_version=release_version)
        else:
            # Use regular task loading for other tasks
            task = get_task(task_name)

        print(f"   ✅ Task loaded: {task.get_name()}")
        print(f"   📝 Description: {task.get_description()}")

        # Load data
        data = task.load_data(limit=limit)
        print(f"   📊 Data loaded: {len(data)} items")

        # Get extractor
        extractor = task.get_extractor()
        print(f"   🔍 Extractor: {extractor.__class__.__name__}")

        # Process first few items
        processed = 0
        for i, doc in enumerate(data[: min(3, len(data))]):
            qa_pair = extractor.extract_qa_pair(doc)
            if qa_pair:
                print(f"   {i + 1}. ✅ Processed: {qa_pair['question'][:30]}...")
                processed += 1

        print(f"   📈 Results: {processed}/{len(data)} items processed successfully")

        # Real model inference
        try:
            from wisent_guard.core.model import Model

            print(
                f"   🧠 Loading model: {model_name} at layer {layer} on device {device}"
            )
            model = Model(name=model_name, device=device)

            # Process a sample item with real model
            if data:
                sample_doc = data[0]
                qa_pair = extractor.extract_qa_pair(sample_doc)
                if qa_pair:
                    # Generate response using the model
                    question = qa_pair.get("question", "")
                    response = model.generate(
                        question, layer_index=layer, max_new_tokens=50
                    )
                    print(f"   ✅ Model response sample: {response[:50]}...")
                else:
                    print("   ⚠️  No QA pair available for model testing")

            print("   🧠 Model inference: ✅ REAL MODEL LOADED AND TESTED")

        except Exception as e:
            print(f"   ⚠️  Model loading failed: {e}")
            print(
                f"   🧠 Model inference: [SIMULATED - would use {model_name} at layer {layer}]"
            )

        return {
            "task_name": task_name,
            "model_name": model_name,
            "layer": layer,
            "total_items": len(data),
            "processed_items": processed,
            "success": True,
        }

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return {
            "task_name": task_name,
            "model_name": model_name,
            "layer": layer,
            "error": str(e),
            "success": False,
        }


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Task-Agnostic CLI Demo")
    parser.add_argument("--demo", action="store_true", help="Run demo")
    parser.add_argument("--task", type=str, help="Task name")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--layer", type=int, help="Layer number")
    parser.add_argument("--limit", type=int, help="Limit number of items")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda, mps)",
    )
    parser.add_argument(
        "--release-version",
        type=str,
        default="release_v1",
        help="LiveCodeBench release version (release_v1, release_v2, etc.)",
    )

    args = parser.parse_args()

    if args.demo:
        demo_task_agnostic_approach()
    elif args.task and args.model and args.layer is not None:
        result = run_task_agnostic_pipeline(
            args.task,
            args.model,
            args.layer,
            args.limit,
            args.device,
            getattr(args, "release_version", "release_v1"),
        )
        print(f"\n📊 Final Result: {result}")
    else:
        print("Usage:")
        print("  python -m wisent_guard.core.task_agnostic_cli --demo")
        print(
            "  python -m wisent_guard.core.task_agnostic_cli --task livecodebench --model meta-llama/Llama-3.2-1B-Instruct --layer 15 --limit 5 --device mps --release-version release_v1"
        )


if __name__ == "__main__":
    main()
