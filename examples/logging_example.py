"""
Example demonstrating the logging infrastructure for production debuggability.

This example shows how to:
1. Configure logging with different levels and outputs
2. Use structured logging with context
3. Capture errors with full stacktraces
4. Track performance with context managers
5. Monitor training progress with detailed logging
"""

import logging
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import logging utilities
from numpy_dl.utils.logging import (
    configure_logging,
    get_logger,
    ContextLogger,
    log_function_call
)

def basic_logging_example():
    """Demonstrate basic logging with context."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Logging with Context")
    print("="*60)

    # Configure logging
    configure_logging(
        level=logging.INFO,
        log_dir='./logs',
        console=True,
        structured=False  # Human-readable format
    )

    # Get a logger
    logger = get_logger('example')

    # Log with context
    logger.info("Application started", version="1.0.0", environment="development")
    logger.debug("Debug message - won't show at INFO level")
    logger.warning("Warning message", threshold=0.9, current_value=0.95)
    logger.error("Error message", error_code=500, module="auth")


def structured_logging_example():
    """Demonstrate structured (JSON) logging."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Structured Logging (JSON)")
    print("="*60)

    # Reconfigure with JSON output
    configure_logging(
        level=logging.INFO,
        console=True,
        structured=True  # JSON format for parsing
    )

    logger = get_logger('structured')
    logger.info(
        "Model training started",
        model="ResNet50",
        dataset="ImageNet",
        batch_size=32,
        learning_rate=0.001
    )


def error_handling_example():
    """Demonstrate error logging with stacktraces."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Error Handling with Stacktraces")
    print("="*60)

    configure_logging(level=logging.DEBUG, console=True)
    logger = get_logger('errors')

    try:
        # Simulate an error
        result = 10 / 0
    except ZeroDivisionError as e:
        # Log exception with full stacktrace
        logger.exception(
            "Division error occurred",
            operation="division",
            numerator=10,
            denominator=0
        )


def performance_tracking_example():
    """Demonstrate performance tracking with context managers."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Performance Tracking")
    print("="*60)

    import time
    configure_logging(level=logging.DEBUG, console=True)
    logger = get_logger('performance')

    # Use context manager for automatic timing
    with ContextLogger(logger, "data_processing", batch_size=100):
        time.sleep(0.1)  # Simulate processing
        logger.info("Processing data...")

    # Manual performance logging
    start = time.time()
    time.sleep(0.05)
    duration = time.time() - start
    logger.log_performance("model_inference", duration, batch_size=32, device="cpu")


@log_function_call(get_logger('decorated'))
def decorated_function(x, y):
    """Example function with automatic logging decorator."""
    import time
    time.sleep(0.01)
    return x + y


def function_decorator_example():
    """Demonstrate automatic function logging."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Function Decorator for Automatic Logging")
    print("="*60)

    configure_logging(level=logging.DEBUG, console=True)

    # Function calls are automatically logged
    result = decorated_function(10, 20)
    print(f"Result: {result}")


def training_simulation_example():
    """Simulate a training loop with comprehensive logging."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Training Loop Logging Simulation")
    print("="*60)

    import numpy as np

    configure_logging(level=logging.INFO, console=True)
    logger = get_logger('training')

    logger.info(
        "Training started",
        model="CNN",
        epochs=3,
        batch_size=32,
        learning_rate=0.001
    )

    for epoch in range(3):
        logger.info(f"Epoch {epoch + 1}/3 started")

        # Simulate training batches
        for batch in range(5):
            # Simulate loss calculation
            loss = np.random.rand() * (1.0 - epoch * 0.3)

            if batch % 2 == 0:
                logger.debug(
                    "Batch processed",
                    epoch=epoch + 1,
                    batch=batch + 1,
                    loss=float(loss)
                )

            # Simulate gradient issues
            if epoch == 1 and batch == 3:
                logger.warning(
                    "Vanishing gradients detected",
                    epoch=epoch + 1,
                    batch=batch + 1,
                    min_grad_norm=1e-8
                )

        logger.info(
            "Epoch completed",
            epoch=epoch + 1,
            avg_loss=float(np.random.rand() * (1.0 - epoch * 0.3))
        )

    logger.info(
        "Training completed successfully",
        total_epochs=3,
        final_loss=0.15
    )


def main():
    """Run all logging examples."""
    print("\n" + "#"*60)
    print("# NumPy Deep Learning - Logging Examples")
    print("# Demonstrating production-ready logging capabilities")
    print("#"*60)

    try:
        basic_logging_example()
        structured_logging_example()
        error_handling_example()
        performance_tracking_example()
        function_decorator_example()
        training_simulation_example()

        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nLogging features demonstrated:")
        print("  ✓ Structured logging with context")
        print("  ✓ Error handling with stacktraces")
        print("  ✓ Performance tracking")
        print("  ✓ Automatic function logging")
        print("  ✓ Training loop logging")
        print("\nLog files saved to: ./logs/")
        print("="*60)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
