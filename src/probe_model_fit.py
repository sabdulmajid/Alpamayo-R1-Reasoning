"""
Probe script to test if Alpamayo-R1-10B fits on A4500 (20GB) in bfloat16.

Uses the official alpamayo_r1 package loading approach.
"""

import torch
import gc
import os
import sys

# Bypass NVML-dependent caching allocator (driver/library mismatch on compute node)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "0")

# Ensure we're using the right cache
os.environ["HF_HOME"] = "/mnt/slurm_nfs/a6abdulm/.cache/huggingface"


def print_gpu_memory(label: str = ""):
    """Print current GPU memory state."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - reserved
        print(f"[VRAM {label}] Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Free: {free:.2f}GB | Total: {total:.2f}GB")
    else:
        print("[VRAM] CUDA not available")


def probe_model_loading():
    """
    Attempt to load Alpamayo-R1-10B in bfloat16 and report results.
    Uses the official AlpamayoR1.from_pretrained() method.
    """
    print("=" * 60)
    print("ALPAMAYO-R1-10B BFLOAT16 LOADING PROBE")
    print("Using official alpamayo_r1 package")
    print("=" * 60)
    print()
    
    # Step 1: Initial state
    print("STEP 1: Initial GPU State")
    print_gpu_memory("initial")
    print()
    
    # Step 2: Check CUDA availability
    print("STEP 2: Checking CUDA...")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  Total memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print()
    
    # Step 3: CUDA warmup — verify allocator works before loading 20GB model
    print("STEP 3: CUDA allocator warmup...")
    try:
        warmup = torch.zeros(1, device="cuda", dtype=torch.bfloat16)
        del warmup
        print("  CUDA allocator OK")
    except RuntimeError as e:
        print(f"  CUDA allocator FAILED: {e}")
        print("  Cannot proceed without working GPU allocator.")
        return False
    print()

    # Step 4: Import alpamayo_r1
    print("STEP 4: Importing alpamayo_r1...")
    try:
        from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
        from alpamayo_r1 import helper
        print("  alpamayo_r1 imported successfully")
    except ImportError as e:
        print(f"  Failed to import alpamayo_r1: {e}")
        return False
    print_gpu_memory("post-import")
    print()
    
    # Step 5: Load model with device_map="auto" (GPU + CPU offloading)
    model_id = "nvidia/Alpamayo-R1-10B"
    print(f"STEP 5: Loading {model_id} with device_map='auto' (bf16, GPU+CPU offloading)...")
    print("  This will download ~22GB of weights on first run...")
    print()
    
    try:
        model = AlpamayoR1.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="auto",
            max_memory={0: "18GiB", "cpu": "8GiB"},
        )
        print("  Model loaded with auto device placement")
        print_gpu_memory("post-load")
        
        # Report where layers ended up
        device_map = getattr(model, "hf_device_map", {})
        if device_map:
            gpu_layers = [k for k, v in device_map.items() if str(v).startswith("cuda") or v == 0]
            cpu_layers = [k for k, v in device_map.items() if v == "cpu"]
            print(f"  GPU layers: {len(gpu_layers)} | CPU layers: {len(cpu_layers)}")
            if cpu_layers:
                print(f"  CPU-offloaded: {cpu_layers[:5]}{'...' if len(cpu_layers) > 5 else ''}")
        else:
            print("  No device map available (model may be fully on one device)")
        
    except Exception as e:
        print(f"  Load failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 6: Get processor/tokenizer
    print()
    print("STEP 6: Getting processor...")
    try:
        processor = helper.get_processor(model.tokenizer)
        print(f"  Processor loaded: {type(processor)}")
    except Exception as e:
        print(f"  Processor load failed: {e}")
    
    # Step 7: Test inference with text only (no images)
    print()
    print("STEP 7: Testing text-only inference...")
    
    try:
        test_prompt = "The ego vehicle is approaching a red traffic light. Describe what action should be taken."
        
        inputs = model.tokenizer(test_prompt, return_tensors="pt")
        # Send inputs to the same device as the model's first parameter
        first_device = next(model.parameters()).device
        inputs = {k: v.to(first_device) for k, v in inputs.items()}
        
        print(f"  Input tokens: {inputs['input_ids'].shape}")
        print_gpu_memory("pre-inference")
        
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model.vlm.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    pad_token_id=model.tokenizer.pad_token_id or 0
                )
        
        print_gpu_memory("post-inference")
        
        generated_text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  Generated ({len(outputs[0])} tokens):")
        print(f"    {generated_text[:300]}...")
        print()
        print("  TEXT INFERENCE SUCCESS")
        
    except Exception as e:
        print(f"  Text inference failed: {e}")
        import traceback
        traceback.print_exc()
        print("  (This may be expected - full inference requires images)")
    
    print()
    print("=" * 60)
    print("PROBE COMPLETE: Model loaded successfully in bfloat16")
    print("=" * 60)
    
    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return True


if __name__ == "__main__":
    success = probe_model_loading()
    sys.exit(0 if success else 1)
