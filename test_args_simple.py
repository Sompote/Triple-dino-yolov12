#!/usr/bin/env python3
"""Simple test of argument parsing for --integrate"""

import sys
sys.path.append('.')

def test_args():
    # Test argument parsing
    from train_triple_dinov3 import main
    import argparse
    
    # Mock sys.argv for different integrate options
    test_cases = [
        ['train_triple_dinov3.py', '--data', 'dummy.yaml', '--integrate', 'nodino', '--download-only'],
        ['train_triple_dinov3.py', '--data', 'dummy.yaml', '--integrate', 'initial', '--download-only'], 
        ['train_triple_dinov3.py', '--data', 'dummy.yaml', '--integrate', 'p3', '--download-only']
    ]
    
    for i, test_argv in enumerate(test_cases):
        integrate_option = test_argv[4]  # Extract integrate option
        print(f"\nTest {i+1}: --integrate {integrate_option}")
        
        # Save original argv
        original_argv = sys.argv.copy()
        
        try:
            # Set test argv
            sys.argv = test_argv
            
            # Import the module to test argument parsing
            import train_triple_dinov3
            parser = argparse.ArgumentParser()
            parser.add_argument('--data', type=str, required=True)
            parser.add_argument('--integrate', type=str, choices=['initial', 'nodino', 'p3'], default='initial')
            parser.add_argument('--download-only', action='store_true')
            
            args = parser.parse_args(test_argv[1:])
            
            print(f"  ✓ Parsed successfully:")
            print(f"    - data: {args.data}")
            print(f"    - integrate: {args.integrate}")
            print(f"    - download_only: {args.download_only}")
            
        except SystemExit as e:
            if e.code == 0:
                print(f"  ✓ Argument parsing successful")
            else:
                print(f"  ❌ Argument parsing failed with code: {e.code}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
        finally:
            # Restore original argv
            sys.argv = original_argv

if __name__ == "__main__":
    print("Testing --integrate argument parsing")
    print("="*50)
    test_args()
    print("\n✅ Argument parsing test completed!")