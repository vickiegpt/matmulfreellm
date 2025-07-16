import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import mmfreelm
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # Check if user wants to use fixed-point implementation
    use_fixed_point = "--fixed-point" in sys.argv
    
    if use_fixed_point:
        print("Using fixed-point HGRN implementation with ternary_matmul operations...")
        # Import and use the integrated fixed-point version
        from generate_integrated import replace_with_fixed_point_hgrn
        
        # Load model
        name = 'ridger/MMfreeLM-2.7B'
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name).cuda().half()
        
        # Replace HGRN layers with fixed-point implementation
        model = replace_with_fixed_point_hgrn(model)
        print("✓ HGRN layers replaced with fixed-point implementation using ternary_matmul")
    else:
        print("Using standard floating-point implementation...")
        # Original implementation
        name = 'ridger/MMfreeLM-2.7B'
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name).cuda().half()
    
    # Generate text
    input_prompt = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, "
    input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.cuda()
    outputs = model.generate(input_ids, max_length=32, do_sample=True, top_p=0.4, temperature=0.6)
    
    print(f"\nInput: {input_prompt}")
    print(f"Output: {tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]}")
    
    if use_fixed_point:
        print("\n✓ Text generated using fixed-point HGRN with ternary_matmul operations!")
        print("  Each HGRN cell performed 4 ternary_matmul operations:")
        print("  1. Input gate projection (w_i)")
        print("  2. Forget gate projection (w_f)")
        print("  3. Gate projection (w_g)")
        print("  4. Output projection (w_o)")

if __name__ == "__main__":
    main()