"""Test multi-task learning imports and basic functionality."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Test imports
print("Testing multi-task learning imports...")

try:
    # Test loss imports
    from numpy_dl.loss.multitask import (
        MultiTaskLoss,
        UncertaintyWeighting,
        GradNorm,
        DynamicWeightAverage,
    )
    print("✓ Loss functions imported successfully")

    # Test model imports
    from numpy_dl.models.multitask import (
        TaskHead,
        HardParameterSharing,
        SoftParameterSharing,
        MultiTaskModel,
        create_hard_sharing_model,
        create_soft_sharing_model,
    )
    print("✓ Model architectures imported successfully")

    # Test utility imports
    from numpy_dl.utils.multitask import (
        MultiTaskMetrics,
        MultiTaskTrainer,
        create_multitask_dataloader,
    )
    print("✓ Utilities imported successfully")

    # Test top-level imports
    from numpy_dl.loss import (
        MultiTaskLoss as MTL,
        UncertaintyWeighting as UW,
    )
    from numpy_dl.models import (
        TaskHead as TH,
        HardParameterSharing as HPS,
    )
    from numpy_dl.utils import (
        MultiTaskMetrics as MTM,
        MultiTaskTrainer as MTT,
    )
    print("✓ Top-level imports working correctly")

    print("\n" + "=" * 60)
    print("ALL IMPORTS SUCCESSFUL!")
    print("=" * 60)
    print("\nMulti-task learning modules are properly integrated.")
    print("\nAvailable components:")
    print("  - Loss weighting: MultiTaskLoss, UncertaintyWeighting, GradNorm, DWA")
    print("  - Architectures: HardParameterSharing, SoftParameterSharing")
    print("  - Utilities: MultiTaskTrainer, MultiTaskMetrics")

    sys.exit(0)

except ImportError as e:
    print(f"\n✗ Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

except Exception as e:
    print(f"\n✗ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
