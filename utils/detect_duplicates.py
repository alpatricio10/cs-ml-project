"""
Duplicate Image Detection Script
Detects exact and near-duplicate images between your competition dataset and CUB-200

Usage:
    python detect_duplicates.py

This will:
1. Hash all images in your train/val datasets
2. Hash all images in CUB-200 external dataset
3. Find matches
4. Create a clean CUB dataset with duplicates removed
"""

import os
import numpy as np
import cv2
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
from collections import defaultdict
import hashlib

def compute_image_hash(image_path, hash_size=16):
    """
    Compute perceptual hash of an image
    
    This detects:
    - Exact duplicates
    - Near duplicates (slight crops, resizes, compression)
    
    Args:
        image_path: Path to image
        hash_size: Hash resolution (higher = more strict)
    
    Returns:
        Hash string
    """
    try:
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize to hash_size x hash_size
        resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
        
        # Compute mean
        mean = resized.mean()
        
        # Create binary hash
        hash_binary = resized > mean
        
        # Convert to hex string
        hash_str = ''.join(['1' if bit else '0' for bit in hash_binary.flatten()])
        
        return hash_str
    except Exception as e:
        return None

def compute_md5_hash(image_path):
    """
    Compute MD5 hash for exact duplicate detection
    """
    try:
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return None

def hamming_distance(hash1, hash2):
    """
    Calculate hamming distance between two hash strings
    Lower distance = more similar
    """
    if hash1 is None or hash2 is None:
        return float('inf')
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

def find_duplicates(competition_dir, external_dir, output_csv='duplicate_report.csv'):
    """
    Find duplicate images between competition and external datasets
    """
    print("="*70)
    print("DUPLICATE IMAGE DETECTION")
    print("="*70)
    print()
    
    # Configuration
    competition_dirs = {
        'train': os.path.join(competition_dir, 'train_images'),
        'val': os.path.join(competition_dir, 'val_images'),
    }
    
    # Step 1: Hash competition images
    print("Step 1: Hashing competition dataset images...")
    competition_hashes = {}
    competition_md5 = {}
    
    for split_name, split_dir in competition_dirs.items():
        if not os.path.exists(split_dir):
            print(f"⚠️  Directory not found: {split_dir}")
            continue
            
        image_files = list(Path(split_dir).rglob('*.jpg')) + \
                     list(Path(split_dir).rglob('*.jpeg')) + \
                     list(Path(split_dir).rglob('*.png'))
        
        print(f"\nProcessing {split_name} set: {len(image_files)} images")
        
        for img_path in tqdm(image_files, desc=f"Hashing {split_name}"):
            perceptual_hash = compute_image_hash(img_path)
            md5_hash = compute_md5_hash(img_path)
            
            if perceptual_hash:
                competition_hashes[str(img_path)] = perceptual_hash
                competition_md5[str(img_path)] = md5_hash
    
    print(f"\n✓ Hashed {len(competition_hashes)} competition images")
    
    # Step 2: Hash external (CUB) images
    print("\nStep 2: Hashing CUB-200 external images...")
    
    if not os.path.exists(external_dir):
        print(f"❌ External directory not found: {external_dir}")
        print("Please download CUB-200 first!")
        return None
    
    external_files = list(Path(external_dir).rglob('*.jpg')) + \
                    list(Path(external_dir).rglob('*.jpeg')) + \
                    list(Path(external_dir).rglob('*.png'))
    
    print(f"Found {len(external_files)} external images")
    
    external_hashes = {}
    external_md5 = {}
    
    for img_path in tqdm(external_files, desc="Hashing CUB images"):
        perceptual_hash = compute_image_hash(img_path)
        md5_hash = compute_md5_hash(img_path)
        
        if perceptual_hash:
            external_hashes[str(img_path)] = perceptual_hash
            external_md5[str(img_path)] = md5_hash
    
    print(f"✓ Hashed {len(external_hashes)} external images")
    
    # Step 3: Find exact duplicates (MD5)
    print("\nStep 3: Finding exact duplicates (MD5 hash)...")
    
    exact_duplicates = []
    competition_md5_set = set(competition_md5.values())
    
    for ext_path, ext_md5 in external_md5.items():
        if ext_md5 in competition_md5_set:
            # Find which competition image matches
            for comp_path, comp_md5 in competition_md5.items():
                if comp_md5 == ext_md5:
                    exact_duplicates.append({
                        'external_image': ext_path,
                        'competition_image': comp_path,
                        'type': 'exact',
                        'distance': 0
                    })
                    break
    
    print(f"Found {len(exact_duplicates)} exact duplicates")
    
    # Step 4: Find near-duplicates (perceptual hash)
    print("\nStep 4: Finding near-duplicates (perceptual hash)...")
    
    near_duplicates = []
    threshold = 10  # Hamming distance threshold (lower = stricter)
    
    for ext_path, ext_hash in tqdm(external_hashes.items(), desc="Checking similarity"):
        for comp_path, comp_hash in competition_hashes.items():
            distance = hamming_distance(ext_hash, comp_hash)
            
            if distance <= threshold and distance > 0:  # Near duplicate but not exact
                near_duplicates.append({
                    'external_image': ext_path,
                    'competition_image': comp_path,
                    'type': 'near',
                    'distance': distance
                })
    
    print(f"Found {len(near_duplicates)} near-duplicates")
    
    # Step 5: Create report
    all_duplicates = exact_duplicates + near_duplicates
    
    if len(all_duplicates) > 0:
        df = pd.DataFrame(all_duplicates)
        df.to_csv(output_csv, index=False)
        print(f"\n✓ Duplicate report saved to: {output_csv}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Competition images scanned: {len(competition_hashes)}")
    print(f"External images scanned: {len(external_hashes)}")
    print(f"Exact duplicates found: {len(exact_duplicates)}")
    print(f"Near-duplicates found: {len(near_duplicates)}")
    print(f"Total duplicates: {len(all_duplicates)}")
    
    if len(all_duplicates) > 0:
        overlap_rate = (len(all_duplicates) / len(external_hashes)) * 100
        print(f"\nOverlap rate: {overlap_rate:.2f}%")
        
        # Get list of duplicate external images to remove
        duplicate_external_files = set([d['external_image'] for d in all_duplicates])
        
        return {
            'duplicates': all_duplicates,
            'duplicate_files': duplicate_external_files,
            'stats': {
                'total_competition': len(competition_hashes),
                'total_external': len(external_hashes),
                'exact_duplicates': len(exact_duplicates),
                'near_duplicates': len(near_duplicates),
                'overlap_rate': overlap_rate
            }
        }
    else:
        print("\n✅ NO DUPLICATES FOUND!")
        print("Your competition dataset and CUB-200 have no overlap.")
        print("Safe to use CUB-200 for pseudo-labeling!")
        return {
            'duplicates': [],
            'duplicate_files': set(),
            'stats': {
                'total_competition': len(competition_hashes),
                'total_external': len(external_hashes),
                'exact_duplicates': 0,
                'near_duplicates': 0,
                'overlap_rate': 0.0
            }
        }

def create_clean_external_dataset(external_dir, duplicate_files, output_dir='external_birds_clean'):
    """
    Create a clean version of CUB dataset with duplicates removed
    """
    print("\n" + "="*70)
    print("CREATING CLEAN EXTERNAL DATASET")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    external_files = list(Path(external_dir).rglob('*.jpg')) + \
                    list(Path(external_dir).rglob('*.jpeg')) + \
                    list(Path(external_dir).rglob('*.png'))
    
    copied = 0
    skipped = 0
    
    for img_path in tqdm(external_files, desc="Copying clean images"):
        if str(img_path) not in duplicate_files:
            # Copy to clean directory
            import shutil
            dest = os.path.join(output_dir, img_path.name)
            shutil.copy2(img_path, dest)
            copied += 1
        else:
            skipped += 1
    
    print(f"\n✓ Clean dataset created!")
    print(f"  Copied: {copied} images")
    print(f"  Skipped (duplicates): {skipped} images")
    print(f"  Output: {output_dir}/")
    print(f"\nUse this clean directory for pseudo-labeling in Phase 3!")

if __name__ == "__main__":
    # Configuration
    COMPETITION_DIR = 'data'
    EXTERNAL_DIR = 'data/FINAL'
    
    print("Starting duplicate detection...")
    print(f"Competition dataset: {COMPETITION_DIR}")
    print(f"External dataset: {EXTERNAL_DIR}")
    print()
    
    # Find duplicates
    result = find_duplicates(COMPETITION_DIR, EXTERNAL_DIR)
    
    if result and len(result['duplicate_files']) > 0:
        print("\n⚠️  DUPLICATES FOUND!")
        print("\nOptions:")
        print("1. Create clean CUB dataset (recommended)")
        print("2. Skip external data entirely")
        
        response = input("\nCreate clean dataset? (y/n): ")
        
        if response.lower() == 'y':
            create_clean_external_dataset(
                EXTERNAL_DIR, 
                result['duplicate_files'],
                output_dir='external_birds_clean'
            )
            print("\n✓ Update your Phase 3 config to use 'external_birds_clean' directory!")
        else:
            print("\nSkipping clean dataset creation.")
            print("You can manually review duplicates in: duplicate_report.csv")
    else:
        print("\n✅ All clear! No duplicates found.")
        print("Safe to use CUB-200 for pseudo-labeling!")
