#!/usr/bin/env python3
"""
Export trained model weights to C++ header file for easy integration.
"""

import torch
import numpy as np
import argparse
import os

def export_weights_to_cpp_header(model_path, scaler_path, output_header_path):
    """Export PyTorch model weights to C++ header file"""
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model_state = checkpoint['model_state_dict']
    scaler = checkpoint.get('scaler')
    
    # Load scaler separately if needed
    if scaler is None and os.path.exists(scaler_path):
        import pickle
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    
    if scaler is None:
        raise ValueError("Scaler not found in checkpoint or file")
    
    # Extract weights from model state dict
    # Based on SpeedupPredictor architecture
    
    # Non-linear branch
    w1 = model_state['nonlin_branch.0.weight'].cpu().numpy()
    b1 = model_state['nonlin_branch.0.bias'].cpu().numpy()
    w2 = model_state['nonlin_branch.3.weight'].cpu().numpy()
    b2 = model_state['nonlin_branch.3.bias'].cpu().numpy()
    w3 = model_state['nonlin_branch.6.weight'].cpu().numpy()
    b3 = model_state['nonlin_branch.6.bias'].cpu().numpy()
    
    # Linear branch
    linear_weight = model_state['linear_branch.weight'].cpu().numpy().flatten()[0]
    
    # Output layers
    w_out1 = model_state['output.0.weight'].cpu().numpy()
    b_out1 = model_state['output.0.bias'].cpu().numpy()
    w_out2 = model_state['output.2.weight'].cpu().numpy()
    b_out2 = model_state['output.2.bias'].cpu().numpy()
    
    # Scaler parameters
    feature_mean = scaler.mean_.astype(np.float32)
    feature_scale = scaler.scale_.astype(np.float32)
    
    # Generate C++ header file
    with open(output_header_path, 'w') as f:
        f.write("""// SpeedupModelWeights.h
// Auto-generated from trained PyTorch model
// DO NOT EDIT - This file is auto-generated

#ifndef SPEEDUP_MODEL_WEIGHTS_H
#define SPEEDUP_MODEL_WEIGHTS_H

#include <vector>
#include <cstdint>

namespace SpeedupModelWeights {

""")
        
        # Write weights
        f.write(f"// Layer 1: Input ({w1.shape[1]}) -> Hidden1 ({w1.shape[0]})\n")
        f.write("const float W1[{}][{}] = {{\n".format(w1.shape[0], w1.shape[1]))
        for i, row in enumerate(w1):
            f.write("  {")
            f.write(", ".join(f"{x:.8f}f" for x in row))
            f.write("}" + ("," if i < len(w1) - 1 else "") + "\n")
        f.write("};\n\n")
        
        f.write(f"const float B1[{len(b1)}] = {{")
        f.write(", ".join(f"{x:.8f}f" for x in b1))
        f.write("};\n\n")
        
        f.write(f"// Layer 2: Hidden1 ({w2.shape[0]}) -> Hidden2 ({w2.shape[1]})\n")
        f.write("const float W2[{}][{}] = {{\n".format(w2.shape[0], w2.shape[1]))
        for i, row in enumerate(w2):
            f.write("  {")
            f.write(", ".join(f"{x:.8f}f" for x in row))
            f.write("}" + ("," if i < len(w2) - 1 else "") + "\n")
        f.write("};\n\n")
        
        f.write(f"const float B2[{len(b2)}] = {{")
        f.write(", ".join(f"{x:.8f}f" for x in b2))
        f.write("};\n\n")
        
        f.write(f"// Layer 3: Hidden2 ({w3.shape[0]}) -> Hidden3 ({w3.shape[1]})\n")
        f.write("const float W3[{}][{}] = {{\n".format(w3.shape[0], w3.shape[1]))
        for i, row in enumerate(w3):
            f.write("  {")
            f.write(", ".join(f"{x:.8f}f" for x in row))
            f.write("}" + ("," if i < len(w3) - 1 else "") + "\n")
        f.write("};\n\n")
        
        f.write(f"const float B3[{len(b3)}] = {{")
        f.write(", ".join(f"{x:.8f}f" for x in b3))
        f.write("};\n\n")
        
        f.write(f"// Linear branch weight for X_times_Y\n")
        f.write(f"const float LINEAR_WEIGHT = {linear_weight:.8f}f;\n\n")
        
        f.write(f"// Output Layer 1: Combined ({w_out1.shape[0]}) -> Output1 ({w_out1.shape[1]})\n")
        f.write("const float W_OUT1[{}][{}] = {{\n".format(w_out1.shape[0], w_out1.shape[1]))
        for i, row in enumerate(w_out1):
            f.write("  {")
            f.write(", ".join(f"{x:.8f}f" for x in row))
            f.write("}" + ("," if i < len(w_out1) - 1 else "") + "\n")
        f.write("};\n\n")
        
        f.write(f"const float B_OUT1[{len(b_out1)}] = {{")
        f.write(", ".join(f"{x:.8f}f" for x in b_out1))
        f.write("};\n\n")
        
        f.write(f"// Output Layer 2: Output1 ({w_out2.shape[0]}) -> Output ({w_out2.shape[1]})\n")
        f.write("const float W_OUT2[{}][{}] = {{\n".format(w_out2.shape[0], w_out2.shape[1]))
        for i, row in enumerate(w_out2):
            f.write("  {")
            f.write(", ".join(f"{x:.8f}f" for x in row))
            f.write("}" + ("," if i < len(w_out2) - 1 else "") + "\n")
        f.write("};\n\n")
        
        f.write(f"const float B_OUT2[{len(b_out2)}] = {{")
        f.write(", ".join(f"{x:.8f}f" for x in b_out2))
        f.write("};\n\n")
        
        f.write("// Feature normalization\n")
        f.write(f"const float FEATURE_MEAN[{len(feature_mean)}] = {{")
        f.write(", ".join(f"{x:.8f}f" for x in feature_mean))
        f.write("};\n\n")
        
        f.write(f"const float FEATURE_SCALE[{len(feature_scale)}] = {{")
        f.write(", ".join(f"{x:.8f}f" for x in feature_scale))
        f.write("};\n\n")
        
        f.write("}  // namespace SpeedupModelWeights\n\n")
        f.write("#endif  // SPEEDUP_MODEL_WEIGHTS_H\n")
    
    print(f"Weights exported to: {output_header_path}")


def main():
    parser = argparse.ArgumentParser(description='Export model weights to C++ header')
    parser.add_argument('--model', required=True, help='Path to PyTorch model (.pth file)')
    parser.add_argument('--scaler', help='Path to scaler pickle file (if separate)')
    parser.add_argument('--output', required=True, help='Output header file path')
    
    args = parser.parse_args()
    
    export_weights_to_cpp_header(args.model, args.scaler, args.output)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

