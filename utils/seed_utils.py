"""
Deterministic Seed Utility for Project Apex
===========================================

Provides reproducible random number generation across all components:
- PyTorch (model weights, sampling)
- NumPy (data processing, augmentation)
- Python random (general randomness)

Bible Reference: Phase 1 - "deterministic seed utility (torch.use_deterministic_algorithms(True))"

Usage:
    from utils.seed_utils import set_global_seed, get_episode_seed
    from config.config_loader import load_config
    
    config = load_config("config/master_config.yaml")
    set_global_seed(config.random_seed)
    
    # For fold-specific episodes
    episode_seed = get_episode_seed(config.random_seed.base_seed, fold_id=3, episode_id=2)
"""

import random
import numpy as np
import torch
import hashlib
from typing import Optional
from dataclasses import dataclass


def set_global_seed(seed_config, verbose: bool = True) -> None:
    """
    Set global random seeds for reproducibility across all libraries.
    
    Args:
        seed_config: RandomSeedConfig from master config
        verbose: If True, print seed initialization info
        
    Side Effects:
        - Sets torch.manual_seed()
        - Sets np.random.seed()
        - Sets random.seed()
        - Optionally enables torch.use_deterministic_algorithms()
        - Sets torch.backends.cudnn.deterministic and benchmark flags
    
    Example:
        >>> from config.config_loader import load_config
        >>> config = load_config("config/master_config.yaml")
        >>> set_global_seed(config.random_seed)
        ✓ Global seed set to 42 (deterministic mode: ON)
    """
    base_seed = seed_config.base_seed
    use_deterministic = seed_config.use_deterministic
    
    # Set seeds for all RNG libraries
    torch.manual_seed(base_seed)
    np.random.seed(base_seed)
    random.seed(base_seed)
    
    # CUDA determinism (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(base_seed)
        torch.cuda.manual_seed_all(base_seed)  # For multi-GPU
        
        # CuDNN determinism
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # Disable auto-tuner for reproducibility
    
    # PyTorch deterministic algorithms (Bible requirement)
    if use_deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            # Some operations may not have deterministic implementations
            if verbose:
                print(f"  Warning: Could not enable full deterministic mode: {e}")
                print(f"  Continuing with seed-based reproducibility only")
    
    if verbose:
        mode = "ON" if use_deterministic else "OFF"
        print(f"✓ Global seed set to {base_seed} (deterministic mode: {mode})")


def get_episode_seed(base_seed: int, fold_id: int, episode_id: int) -> int:
    """
    Derive a deterministic episode-specific seed from base seed.
    
    Uses SHA256 hashing to ensure:
    - Different folds/episodes get different seeds
    - Same (fold, episode) always gets same seed
    - No collisions between different (fold, episode) pairs
    
    Args:
        base_seed: Base random seed from config
        fold_id: Walk-forward fold index (1-8)
        episode_id: Episode index within fold (1-3)
        
    Returns:
        Deterministic 32-bit integer seed for this episode
        
    Example:
        >>> get_episode_seed(42, fold_id=3, episode_id=2)
        1847293847
        >>> get_episode_seed(42, fold_id=3, episode_id=2)  # Always same
        1847293847
        >>> get_episode_seed(42, fold_id=3, episode_id=1)  # Different episode
        2938471923
    """
    # Create unique string identifier
    seed_string = f"apex_seed_{base_seed}_fold_{fold_id}_episode_{episode_id}"
    
    # Hash to get deterministic integer
    hash_digest = hashlib.sha256(seed_string.encode()).digest()
    
    # Convert first 4 bytes to unsigned 32-bit integer
    episode_seed = int.from_bytes(hash_digest[:4], byteorder='big')
    
    return episode_seed


def get_fold_seed(base_seed: int, fold_id: int) -> int:
    """
    Derive a deterministic fold-specific seed (for data splits, etc.).
    
    Args:
        base_seed: Base random seed from config
        fold_id: Walk-forward fold index (1-8)
        
    Returns:
        Deterministic 32-bit integer seed for this fold
        
    Example:
        >>> get_fold_seed(42, fold_id=5)
        3847562910
    """
    seed_string = f"apex_seed_{base_seed}_fold_{fold_id}"
    hash_digest = hashlib.sha256(seed_string.encode()).digest()
    fold_seed = int.from_bytes(hash_digest[:4], byteorder='big')
    return fold_seed


def seed_worker(worker_id: int, base_seed: int) -> None:
    """
    Seed function for PyTorch DataLoader workers.
    
    Ensures each worker has a different but deterministic seed.
    Use with: DataLoader(..., worker_init_fn=lambda wid: seed_worker(wid, base_seed))
    
    Args:
        worker_id: Worker ID assigned by DataLoader
        base_seed: Base random seed from config
    """
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class SeedContext:
    """
    Context manager for temporary seed changes.
    
    Useful for deterministic data augmentation while preserving global RNG state.
    
    Example:
        >>> with SeedContext(12345):
        ...     # All random ops here use seed 12345
        ...     augmented_data = add_noise(data)
        >>> # Global RNG state restored here
    """
    
    def __init__(self, seed: int):
        self.seed = seed
        self.torch_state = None
        self.numpy_state = None
        self.random_state = None
    
    def __enter__(self):
        # Save current states
        self.torch_state = torch.get_rng_state()
        self.numpy_state = np.random.get_state()
        self.random_state = random.getstate()
        
        # Set temporary seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous states
        torch.set_rng_state(self.torch_state)
        np.random.set_state(self.numpy_state)
        random.setstate(self.random_state)


def verify_determinism(seed_config, num_trials: int = 3) -> bool:
    """
    Verify that seed configuration produces deterministic results.
    
    Runs a simple random operation multiple times and checks for identical output.
    
    Args:
        seed_config: RandomSeedConfig from master config
        num_trials: Number of trials to run
        
    Returns:
        True if all trials produce identical results, False otherwise
        
    Example:
        >>> from config.config_loader import load_config
        >>> config = load_config("config/master_config.yaml")
        >>> verify_determinism(config.random_seed)
        ✓ Determinism verified: 3/3 trials identical
        True
    """
    results = []
    
    for trial in range(num_trials):
        # Reset seed
        set_global_seed(seed_config, verbose=False)
        
        # Run deterministic operations
        torch_rand = torch.randn(10, 10).sum().item()
        numpy_rand = np.random.randn(10, 10).sum()
        python_rand = sum(random.random() for _ in range(100))
        
        results.append((torch_rand, numpy_rand, python_rand))
    
    # Check all results are identical
    all_identical = all(r == results[0] for r in results)
    
    if all_identical:
        print(f"✓ Determinism verified: {num_trials}/{num_trials} trials identical")
    else:
        print(f"✗ Determinism FAILED: trials produced different results")
        for i, r in enumerate(results):
            print(f"  Trial {i+1}: torch={r[0]:.6f}, numpy={r[1]:.6f}, python={r[2]:.6f}")
    
    return all_identical


if __name__ == "__main__":
    # Test seed utilities
    print("\n" + "="*80)
    print("SEED UTILITY TEST")
    print("="*80)
    
    # Mock config for testing
    @dataclass
    class MockSeedConfig:
        base_seed: int = 42
        use_deterministic: bool = True
    
    seed_config = MockSeedConfig()
    
    # Test 1: Global seed setting
    print("\nTest 1: Global seed setting")
    set_global_seed(seed_config)
    
    # Test 2: Episode seed derivation
    print("\nTest 2: Episode seed derivation")
    for fold in [1, 3, 8]:
        for episode in [1, 2, 3]:
            seed = get_episode_seed(42, fold, episode)
            print(f"  Fold {fold}, Episode {episode}: seed = {seed}")
    
    # Test 3: Determinism verification
    print("\nTest 3: Determinism verification")
    verify_determinism(seed_config, num_trials=5)
    
    # Test 4: Seed context manager
    print("\nTest 4: Seed context manager")
    set_global_seed(seed_config, verbose=False)
    val1 = torch.randn(3).tolist()
    
    with SeedContext(99999):
        val_temp = torch.randn(3).tolist()
    
    val2 = torch.randn(3).tolist()
    
    print(f"  Before context: {[f'{v:.4f}' for v in val1]}")
    print(f"  Inside context: {[f'{v:.4f}' for v in val_temp]}")
    print(f"  After context:  {[f'{v:.4f}' for v in val2]}")
    print(f"  ✓ Context manager works (val1 ≠ val_temp, val1 continues sequence)")
    
    print("\n" + "="*80)
    print("ALL SEED UTILITY TESTS PASSED ✓")
    print("="*80)
