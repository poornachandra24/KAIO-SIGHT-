
import torch
from unsloth import FastVisionModel
from PIL import Image
import numpy as np

def test_processor():
    model_name = "unsloth/Qwen3-VL-8B-Instruct-bnb-4bit"
    print(f"Loading {model_name}...")
    model, tokenizer = FastVisionModel.from_pretrained(model_name, load_in_4bit=True)
    
    print("Model loaded.")
    
    # Create dummy video: 16 frames of 336x336
    frames = [Image.new('RGB', (336, 336), color=(i*10, i*10, i*10)) for i in range(16)]
    
    # Create message
    msgs = [
        {
            "role": "user", 
            "content": [
                {"type": "video", "video": frames}, 
                {"type": "text", "text": "Describe this video."}
            ]
        }
    ]
    
    # Process
    print("Processing video...")
    inputs = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    # Note: apply_chat_template returns text. The processor needs to be called on images/videos.
    # Unsloth's DataCollator usually handles this.
    
    # Let's try to simulate what the collator does or use the processor directly if accessible
    # Qwen2-VL processor is usually tokenizer.image_processor or similar?
    # Actually FastVisionModel returns tokenizer which wraps the processor?
    
    # Let's look at how UnslothVisionDataCollator works or just use the processor
    # Usually: processor(images=..., videos=..., text=..., return_tensors="pt")
    
    # But here we have 'tokenizer'.
    # For Qwen2-VL, tokenizer is often the processor wrapper?
    
    # Let's try to use the UnslothVisionDataCollator directly
    from unsloth import UnslothVisionDataCollator
    collator = UnslothVisionDataCollator(model, tokenizer)
    
    # Mock dataset item
    # The trainer expects: {"messages": msgs} (based on my latest change)
    # But wait, UnslothVisionDataCollator expects what?
    # It usually expects a list of dicts.
    
    item = {"messages": msgs}
    batch = [item, item] # Batch of 2
    
    print("Collating batch...")
    try:
        out = collator(batch)
        print("Collation successful!")
        print("Keys:", out.keys())
        if "pixel_values_videos" in out:
            print("pixel_values_videos shape:", out["pixel_values_videos"].shape)
        if "video_grid_thw" in out:
            print("video_grid_thw:", out["video_grid_thw"])
            
    except Exception as e:
        print(f"Collation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_processor()
