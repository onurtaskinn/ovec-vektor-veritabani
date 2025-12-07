#!/usr/bin/env python3
"""
Simple NPZ File Visualizer
Displays the contents of NumPy .npz files in a readable format.

Usage:
    python npz_viewer.py <file.npz>
"""

import sys
import numpy as np
from pathlib import Path


def format_size(num_bytes):
    """Format bytes into human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


def preview_array(arr, max_items=10):
    """Generate a preview of array contents."""
    if arr.dtype == object:
        # Handle object arrays (lists, strings, etc.)
        if len(arr) <= max_items:
            return str(arr)
        else:
            preview = ', '.join(str(arr[i]) for i in range(min(3, len(arr))))
            return f"[{preview}, ... ({len(arr)} items)]"
    
    elif arr.ndim == 1:
        # 1D array
        if len(arr) <= max_items:
            return str(arr)
        else:
            return f"[{arr[0]:.6f}, {arr[1]:.6f}, ..., {arr[-1]:.6f}]"
    
    elif arr.ndim == 2:
        # 2D array - show first row
        if arr.shape[0] == 0:
            return "[]"
        first_row = arr[0]
        if len(first_row) <= max_items:
            preview = ', '.join(f"{x:.6f}" for x in first_row[:max_items])
        else:
            preview = ', '.join(f"{x:.6f}" for x in first_row[:3])
            preview += f", ... ({len(first_row)} values)"
        return f"First row: [{preview}]"
    
    else:
        # Higher dimensional
        return f"Shape: {arr.shape}"


def visualize_npz(filepath):
    """Visualize the contents of an NPZ file."""
    filepath = Path(filepath)
    
    if not filepath.exists():
        print(f"❌ Error: File '{filepath}' not found")
        return
    
    if not filepath.suffix == '.npz':
        print(f"⚠️  Warning: File doesn't have .npz extension")
    
    # Load the NPZ file
    print(f"\n📦 NPZ File: {filepath.name}")
    print(f"📍 Location: {filepath.parent}")
    print(f"💾 Size: {format_size(filepath.stat().st_size)}")
    print("=" * 70)
    
    data = np.load(filepath, allow_pickle=True)
    
    print(f"\n📂 Contents: {len(data.files)} array(s)\n")
    
    total_size = 0
    
    for i, key in enumerate(data.files, 1):
        arr = data[key]
        
        print(f"{i}. 🔹 {key}")
        print(f"   Type:     {type(arr).__name__}")
        print(f"   Dtype:    {arr.dtype}")
        print(f"   Shape:    {arr.shape}")
        
        if hasattr(arr, 'nbytes'):
            size = arr.nbytes
            total_size += size
            print(f"   Size:     {format_size(size)}")
        
        # Show preview
        print(f"   Preview:  {preview_array(arr)}")
        print()
    
    print("=" * 70)
    print(f"Total data size: {format_size(total_size)}")
    
    data.close()


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python npz_viewer.py <file.npz>")
        print("\nExample:")
        print("  python npz_viewer.py vectors.npz")
        print("  python npz_viewer.py examples/real_embeddings/db/ivf_index.npz")
        sys.exit(1)
    
    filepath = sys.argv[1]
    visualize_npz(filepath)


if __name__ == "__main__":
    main()

