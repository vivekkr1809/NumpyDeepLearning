#!/usr/bin/env python
"""Test runner script for NumpyDeepLearning.

This script provides a convenient way to run tests with different configurations.

Usage:
    # Run all tests
    python run_tests.py

    # Run only unit tests
    python run_tests.py --unit

    # Run only functional tests
    python run_tests.py --functional

    # Run only integration tests
    python run_tests.py --integration

    # Run with coverage
    python run_tests.py --coverage

    # Run specific test file
    python run_tests.py tests/unit/test_tensor.py

    # Run with verbose output
    python run_tests.py -v

    # Run fast tests only (skip slow tests)
    python run_tests.py --fast
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_tests(args):
    """Run tests with specified configuration."""
    # Base pytest command
    cmd = [sys.executable, "-m", "pytest"]

    # Add test selection
    if args.unit:
        cmd.extend(["-m", "unit", "tests/unit/"])
    elif args.functional:
        cmd.extend(["-m", "functional", "tests/functional/"])
    elif args.integration:
        cmd.extend(["-m", "integration", "tests/integration/"])
    elif args.test_file:
        cmd.append(args.test_file)
    else:
        # Run all tests
        cmd.append("tests/")

    # Add verbosity
    if args.verbose:
        cmd.append("-vv")
    elif not args.quiet:
        cmd.append("-v")

    # Add coverage
    if args.coverage:
        cmd.extend([
            "--cov=numpy_dl",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])

    # Skip slow tests if requested
    if args.fast:
        cmd.extend(["-m", "not slow"])

    # Fail fast
    if args.failfast:
        cmd.append("-x")

    # Show local variables on failure
    if args.showlocals:
        cmd.append("-l")

    # Add any additional pytest arguments
    if args.pytest_args:
        cmd.extend(args.pytest_args)

    # Print command
    print(f"Running: {' '.join(cmd)}")
    print("-" * 80)

    # Run tests
    result = subprocess.run(cmd)
    return result.returncode


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run tests for NumpyDeepLearning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Test selection
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument(
        "--unit",
        action="store_true",
        help="Run only unit tests"
    )
    test_group.add_argument(
        "--functional",
        action="store_true",
        help="Run only functional tests"
    )
    test_group.add_argument(
        "--integration",
        action="store_true",
        help="Run only integration tests"
    )
    test_group.add_argument(
        "test_file",
        nargs="?",
        help="Specific test file to run"
    )

    # Output options
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    output_group.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet output"
    )

    # Coverage
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run with coverage reporting"
    )

    # Performance
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip slow tests"
    )

    # Debugging
    parser.add_argument(
        "-x", "--failfast",
        action="store_true",
        help="Stop on first failure"
    )
    parser.add_argument(
        "-l", "--showlocals",
        action="store_true",
        help="Show local variables in tracebacks"
    )

    # Additional pytest arguments
    parser.add_argument(
        "pytest_args",
        nargs="*",
        help="Additional arguments to pass to pytest"
    )

    args = parser.parse_args()

    # Check if pytest is available
    try:
        subprocess.run(
            [sys.executable, "-m", "pytest", "--version"],
            check=True,
            capture_output=True
        )
    except subprocess.CalledProcessError:
        print("Error: pytest is not installed.")
        print("Install it with: pip install pytest pytest-cov")
        return 1

    # Run tests
    return run_tests(args)


if __name__ == "__main__":
    sys.exit(main())
