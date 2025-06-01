# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 2024

@author: raxephion
Perchance Revival - Recreating Old Perchance SD 1.5 Experience
Basic Stable Diffusion 1.5 Gradio App with local/Hub models and CPU/GPU selection
Added multi-image generation capability.

NOTE: App still in early development - UI will be adjusted to match Perchance presets
"""

import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
# Import commonly used schedulers
from diffusers import DDPMScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler, LMSDiscreteScheduler
import os
from PIL import Image
import time # Optional: for timing generation
import random # Needed for random seed
import numpy as np # Needed for MAX_SEED, even if not used directly with gr.Number(-1) input
# from huggingface_hub import HfFolder # Uncomment if you need to check for HF token

# --- Configuration ---
MODELS_DIR = "checkpoints" # Directory for user's *additional* local models
# Standard SD 1.5 sizes (multiples of 64 are generally safe)
# Models are primarily trained on 512x512. Other sizes might show artifacts.
# Added 'hire.fix' as an option, interpreted as 1024x1024 in this script
SUPPORTED_SD15_SIZES = ["512x512", "768x512", "512x768", "768x768", "1024x768", "768x1024", "1024x1024", "hire.fix"]

# Mapping of friendly scheduler names to their diffusers classes
SCHEDULER_MAP = {
    "Euler": EulerDiscreteScheduler,
    "DPM++ 2M": DPMSolverMultistepScheduler,
    "DDPM": DDPMScheduler,
    "LMS": LMSDiscreteScheduler,
    # Add more as needed from diffusers.schedulers (make sure they are imported)
}
DEFAULT_SCHEDULER = "Euler" # Default scheduler on startup

# --- Perchance Revival Specific: Map Styles to Models ---
# This dictionary defines the "Styles" shown in the UI dropdown
# and maps them to the Hugging Face Hub ID or local path of the model to load.
# Add more "Style Name": "Model ID/Path" entries here for models you want to feature with a style name
STYLE_MODEL_MAP = {
    "Drawn Anime": "Yntec/RevAnimatedV2Rebirth",
    "Mix Anime": "stablediffusionapi/realcartoon-anime-v11",
    # Add other style mappings here, e.g., "Realistic": "runwayml/stable-diffusion-v1-5",
}

# DEFAULT_HUB_MODELS list is now empty or used for other *non-styled* Hub models
# if you want to list them by their raw Hub ID alongside styles.
# For this specific request, we will rely only on STYLE_MODEL_MAP for featured models.
DEFAULT_HUB_MODELS = [] # Keep empty as styles handle featured models

# ------------------------------------------------------


# --- Constants for UI / Generation ---
MAX_SEED = np.iinfo(np.int32).max # Define MAX_SEED for random number generation

# --- Determine available devices and set up options ---
AVAILABLE_DEVICES = ["CPU"]
if torch.cuda.is_available():
    AVAILABLE_DEVICES.append("GPU")
    print(f"CUDA available. Found {torch.cuda.device_count()} GPU(s).")
    if torch.cuda.device_count() > 0:
        print(f"Using GPU 0: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available. Running on CPU.")

# Default device preference: GPU if available, else CPU
DEFAULT_DEVICE = "GPU" if "GPU" in AVAILABLE_DEVICES else "CPU"


# --- Global state for the loaded pipeline ---
# We'll load the pipeline once and keep it in memory
current_pipeline = None
current_model_id_loaded = None # Keep track of the actual model ID loaded
current_device_loaded = None # Keep track of the device the pipeline is currently on


# --- Helper function to list available local models ---
def list_local_models(models_dir):
    """Scans the specified directory for subdirectories (potential local diffusers models)."""
    # Create the models directory if it doesn't exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created directory: {models_dir}")
        return [] # No models if directory was just created

    # List subdirectories (potential models)
    # Return their full relative path from the script location
    local_models = [os.path.join(models_dir, d) for d in os.listdir(models_dir)
                    if os.path.isdir(os.path.join(models_dir, d))]

    return local_models


# --- Image Generation Function ---
# 'model_input_name' will be the selected Style name or "Local: path"
def generate_image(model_input_name, selected_device_str, prompt, negative_prompt, steps, cfg_scale, scheduler_name, size, seed, num_images):
    """Generates images using the selected model (based on style/path) and parameters on the chosen device."""
    global current_pipeline, current_model_id_loaded, current_device_loaded, SCHEDULER_MAP, MAX_SEED, STYLE_MODEL_MAP

    if not model_input_name or model_input_name == "No models found":
        raise gr.Error("No model/style selected or available. Please select a Style or add local models.")
    if not prompt:
        raise gr.Error("Please enter a prompt.")

    num_images_int = int(num_images) # Convert num_images slider value to int
    if num_images_int <= 0:
         raise gr.Error("Number of images must be at least 1.")

    # Map selected device string to PyTorch device string
    device_to_use = "cuda" if selected_device_str == "GPU" and "GPU" in AVAILABLE_DEVICES else "cpu"
    # If GPU was selected but not available, raise an error specific to this condition
    if selected_device_str == "GPU" and device_to_use == "cpu":
         raise gr.Error("GPU selected but CUDA is not available to PyTorch. Ensure you have a compatible NVIDIA GPU, the correct drivers, and have installed the CUDA version of PyTorch in your environment.")


    # Determine dtype based on the actual device being used
    dtype_to_use = torch.float32 # Default
    if device_to_use == "cuda":
        # Check compute capability (7.0+ for good fp16 on Ampere/Turing+)
        # Also add check for is_available just to be super safe before calling get_device_capability
        if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7:
             dtype_to_use = torch.float16
             print("GPU supports FP16, using torch.float16 for potential performance/memory savings.")
        else:
             dtype_to_use = torch.float32 # Fallback if GPU is very old or check fails
             print("GPU might not fully support FP16 or capability check failed, using torch.float32.")
    else:
         dtype_to_use = torch.float32 # CPU requires float32


    print(f"Attempting generation on device: {device_to_use}, using dtype: {dtype_to_use}")

    # --- 1. Determine the actual model ID/path from the selected input name ---
    actual_model_id_to_load = None

    if model_input_name in STYLE_MODEL_MAP:
        actual_model_id_to_load = STYLE_MODEL_MAP[model_input_name]
        print(f"Selected style '{model_input_name}' maps to model: {actual_model_id_to_load}")
    elif model_input_name.startswith("Local: "):
        # It's a local model selection with the "Local: " prefix
        actual_model_id_to_load = model_input_name.replace("Local: ", "", 1) # Remove prefix
        print(f"Selected local model path: {actual_model_id_to_load}")
    else:
        # Fallback: If somehow an input name isn't in the map or doesn't have the prefix,
        # assume it's a raw ID/path (shouldn't happen with correct UI population)
        actual_model_id_to_load = model_input_name
        print(f"Selected identifier '{model_input_name}' not a style or local path format, attempting to load as raw ID/path...")

    if not actual_model_id_to_load or actual_model_id_to_load == "No models found":
         raise gr.Error("Invalid model selection. Could not determine which model to load.")


    # --- 2. Load Model if necessary ---
    # Check if the requested model ID/path OR the device has changed
    if current_pipeline is None or current_model_id_loaded != actual_model_id_to_load or (current_device_loaded is not None and str(current_device_loaded) != device_to_use):
        print(f"Loading model: {actual_model_id_to_load} onto {device_to_use}...")
        # Clear previous pipeline to potentially free memory *before* loading the new one
        if current_pipeline is not None:
             print(f"Unloading previous model '{current_model_id_loaded}' from {current_device_loaded}...")
             # Move pipeline to CPU before deleting if it was on GPU, might help with freeing VRAM
             if str(current_device_loaded) == "cuda":
                  try:
                      current_pipeline.to("cpu")
                      print("Moved previous pipeline to CPU.")
                  except Exception as move_e:
                      print(f"Warning: Failed to move previous pipeline to CPU: {move_e}")

             del current_pipeline
             current_pipeline = None # Set to None immediately
             # Attempt to clear CUDA cache if using GPU (from the previous device)
             if str(current_device_loaded) == "cuda":
                 try:
                     torch.cuda.empty_cache()
                     print("Cleared CUDA cache.")
                 except Exception as cache_e:
                     print(f"Warning: Error clearing CUDA cache: {cache_e}") # Don't stop if cache clearing fails

        # Ensure the device is actually available if not CPU (redundant with initial check but safe)
        if device_to_use == "cuda":
             if not torch.cuda.is_available():
                  raise gr.Error("CUDA selected but not available. Please install PyTorch with CUDA support or select CPU.")

        try:
            # Load the pipeline
            # Determine if it's likely a local path by checking if it exists as a directory
            is_local_path_check = os.path.isdir(actual_model_id_to_load)

            if is_local_path_check:
                 print(f"Attempting to load local model from: {actual_model_id_to_load}")
                 pipeline = StableDiffusionPipeline.from_pretrained(
                     actual_model_id_to_load,
                     torch_dtype=dtype_to_use,
                     safety_checker=None, # Removed for simplicity/speed, use with caution
                 )
            else:
                 print(f"Attempting to load Hub model: {actual_model_id_to_load}")
                 # Hugging Face Hub models are loaded by their ID
                 pipeline = StableDiffusionPipeline.from_pretrained(
                     actual_model_id_to_load,
                     torch_dtype=dtype_to_use,
                     safety_checker=None, # Removed for simplicity/speed, use with caution
                 )


            pipeline = pipeline.to(device_to_use) # Move to the selected device

            current_pipeline = pipeline
            current_model_id_loaded = actual_model_id_to_load # Store the actual ID/path loaded
            current_device_loaded = torch.device(device_to_use)

            # Basic check for SD1.x architecture (cross_attention_dim = 768)
            unet_config = getattr(pipeline, 'unet', None)
            if unet_config and hasattr(unet_config, 'config') and hasattr(unet_config.config, 'cross_attention_dim'):
                 cross_attn_dim = unet_config.config.cross_attention_dim
                 if cross_attn_dim != 768:
                     warning_msg = (f"Warning: Loaded model '{actual_model_id_to_load}' might not be a standard SD 1.x model "
                                    f"(expected UNet cross_attention_dim 768, found {cross_attn_dim}). "
                                    "Results may be unexpected or generation might fail.")
                     print(warning_msg)
                     gr.Warning(warning_msg)
                 else:
                     print("UNet cross_attention_dim is 768, consistent with SD 1.x.")
            else:
                 print("Could not check UNet cross_attention_dim.")


            print(f"Model '{actual_model_id_to_load}' loaded successfully on {current_device_loaded} with dtype {dtype_to_use}.")

        except Exception as e:
            # Reset global state on load failure
            current_pipeline = None
            current_model_id_loaded = None
            current_device_loaded = None
            print(f"Error loading model '{actual_model_id_to_load}': {e}")
            error_message_lower = str(e).lower()

            # --- Specific error handling for the PyTorch version vulnerability ---
            if "require users to upgrade torch to at least v2.6" in error_message_lower or "vulnerability issue in `torch.load`" in error_message_lower:
                 print("\n--- HINT: PyTorch version likely too old for this model/library version ---")
                 print("The model uses a file format requiring a newer PyTorch.")
                 print("Run setup.bat again or manually install PyTorch 2.6+ (if available) for your system.")
                 print("See https://pytorch.org/get-started/locally/ for commands.")
                 print("------------------------------------------------------------------------\n")
                 raise gr.Error(
                     f"Failed to load model '{actual_model_id_to_load}': Your installed PyTorch version is too old "
                     f"for this model file format. You need PyTorch 2.6 or higher. "
                     f"Run `setup.bat` again, or manually install an updated PyTorch version. "
                     f"See instructions on the PyTorch website: https://pytorch.org/get-started/locally/. Error: {e}"
                 )
            # --- End specific error handling ---

            # Provide more specific error messages based on common exceptions (existing checks)
            elif "cannot find requested files" in error_message_lower or "404 client error" in error_message_lower or "no such file or directory" in error_message_lower:
                 raise gr.Error(f"Model '{actual_model_id_to_load}' not found. Check name/path, Hugging Face Hub ID spelling, or internet connection. Error: {e}")
            elif "checkpointsnotfounderror" in error_message_lower or "valueerror: could not find a valid model structure" in error_message_lower:
                 raise gr.Error(f"No valid diffusers model at '{actual_model_id_to_load}'. Ensure it's a diffusers format directory or a valid Hub ID. Error: {e}")
            elif "out of memory" in error_message_lower:
                 raise gr.Error(f"Out of Memory (OOM) loading model '{actual_model_id_to_load}'. Try a lighter model (e.g., pruned, or less VRAM-hungry) or select CPU. Error: {e}")
            elif "cusolver64" in error_message_lower or "cuda driver version" in error_message_lower or "cuda error" in error_message_lower:
                 raise gr.Error(f"CUDA/GPU Driver Error: {e} loading '{actual_model_id_to_load}'. Check drivers, PyTorch with CUDA installation, or select CPU.")
            elif "safetensors_rust.safetensorserror" in error_message_lower or "oserror: cannot load" in error_message_lower or "filenotfounderror" in error_message_lower:
                 raise gr.Error(f"Model file error for '{actual_model_id_to_load}': {e}. Files might be corrupt, incomplete, or the path is wrong.")
            elif "could not import" in error_message_lower or "module not found" in error_message_lower:
                 raise gr.Error(f"Dependency error: {e} during model loading. Ensure all dependencies are installed (run setup.bat) and PyTorch is installed correctly for your device.")
            else:
                 # Generic catch-all for unknown errors
                raise gr.Error(f"Failed to load model '{actual_model_id_to_load}': An unexpected error occurred. {e}")

    # Check if pipeline is successfully loaded before proceeding
    if current_pipeline is None:
         # This check should ideally be caught by the error handling above, but as a failsafe:
         raise gr.Error(f"Model '{actual_model_id_to_load}' failed to load previously. Cannot generate image.")


    # 3. Configure Scheduler
    selected_scheduler_class = SCHEDULER_MAP.get(scheduler_name)
    if selected_scheduler_class is None:
         print(f"Warning: Unknown scheduler '{scheduler_name}'. Using default: {DEFAULT_SCHEDULER}.")
         selected_scheduler_class = SCHEDULER_MAP[DEFAULT_SCHEDULER]
         gr.Warning(f"Unknown scheduler '{scheduler_name}'. Using default: {DEFAULT_SCHEDULER}.")

    # Recreate scheduler from config to ensure compatibility with the loaded pipeline
    try:
        scheduler_config = current_pipeline.scheduler.config
        current_pipeline.scheduler = selected_scheduler_class.from_config(scheduler_config)
        print(f"Scheduler set to: {scheduler_name}")
    except Exception as e:
        print(f"Error setting scheduler '{scheduler_name}': {e}")
        # Attempt to fallback to a default if setting fails
        try:
             print(f"Attempting to fallback to default scheduler: {DEFAULT_SCHEDULER}")
             current_pipeline.scheduler = SCHEDULER_MAP[DEFAULT_SCHEDULER].from_config(scheduler_config)
             gr.Warning(f"Failed to set scheduler to '{scheduler_name}', fell back to {DEFAULT_SCHEDULER}. Error: {e}")
        except Exception as fallback_e:
             print(f"Fallback scheduler failed too: {fallback_e}")
             raise gr.Error(f"Failed to configure scheduler '{scheduler_name}' and fallback failed. Error: {fallback_e} (Original: {e})") # Include both errors


    # 4. Parse Image Size
    width, height = 512, 512 # Default size
    if size.lower() == "hire.fix":
        width, height = 1024, 1024
        print(f"Interpreting 'hire.fix' size as {width}x{height}")
    else:
        try:
            w_str, h_str = size.split('x')
            width = int(w_str)
            height = int(h_str)
        except ValueError:
            raise gr.Error(f"Invalid size format: '{size}'. Use 'WidthxHeight' (e.g., 512x512) or 'hire.fix'.")
        except Exception as e:
             raise gr.Error(f"Error parsing size '{size}': {e}")

    # Size multiple check (SD 1.5 works best with multiples of 64 or 8)
    # 64 is safer/more common. Check both.
    multiple_check = 64
    if width % multiple_check != 0 or height % multiple_check != 0:
         warning_msg_size = (f"Warning: Image size {width}x{height} is not a multiple of {multiple_check}. "
                             f"Stable Diffusion 1.5 models are typically trained on sizes like 512x512 or 768x768. "
                             "Using non-standard sizes may cause tiling, distortions, or other artifacts.")
         print(warning_msg_size)
         gr.Warning(warning_msg_size)


    # 5. Set Seed Generator
    generator = None
    # The generator device needs to match the pipeline device
    generator_device = current_pipeline.device if current_pipeline else torch.device(device_to_use)

    # Handle seed based on input (-1 for random)
    seed_int = int(seed) # Get the integer value from the input

    if seed_int == -1: # Check if the user explicitly requested a random seed
        seed_int = random.randint(0, MAX_SEED) # Generate a random seed
        print(f"User requested random seed (-1), generated: {seed_int}")
    else:
        print(f"Using provided seed: {seed_int}")


    try:
        # Explicitly move generator to the desired device
        generator = torch.Generator(device=generator_device).manual_seed(seed_int)
        print(f"Generator set with seed {seed_int} on device: {generator_device}")
    except Exception as e:
         print(f"Warning: Error setting seed generator on device {generator_device}: {e}. Falling back to default generator (potentially on CPU) or system random.")
         gr.Warning(f"Failed to set seed generator with seed {seed_int}. Using random seed. Error: {e}")
         generator = None # Let pipeline handle random seed if generator creation fails or device mismatch
         # If generator creation failed, the actual seed used by the pipeline will be different and system-dependent random.
         # We should probably report -1 in this case, or just report the seed we tried to use.
         # Reporting the seed we *tried* to use is simpler and often sufficient.
         pass # Keep the last calculated seed_int


    # 6. Generate Images
    num_images_int = int(num_images) # Convert num_images slider value to int

    # Log which style/model was used for this generation request
    print(f"Generating {num_images_int} image(s) for Style/Model '{model_input_name}' (Actual Model: {actual_model_id_to_load}): Prompt='{prompt[:80]}{'...' if len(prompt) > 80 else ''}', NegPrompt='{negative_prompt[:80]}{'...' if len(negative_prompt) > 80 else ''}', Steps={int(steps)}, CFG={float(cfg_scale)}, Size={width}x{height}, Scheduler={scheduler_name}, Seed={seed_int if generator else 'System Random'}, Images={num_images_int}")
    start_time = time.time()

    try:
        # Ensure required parameters are integers/floats
        num_inference_steps_int = int(steps)
        guidance_scale_float = float(cfg_scale)

        # Basic validation on parameters
        if num_inference_steps_int <= 0 or guidance_scale_float <= 0:
             raise ValueError("Steps and CFG Scale must be positive values.")
        if width <= 0 or height <= 0:
             raise ValueError("Image width and height must be positive.")


        output = current_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_inference_steps=num_inference_steps_int,
            guidance_scale=guidance_scale_float,
            width=width,
            height=height,
            generator=generator,
            num_images_per_prompt=num_images_int, # <-- Pass the number of images here
            # Add VAE usage here if needed for specific models that require it
            # vae=...
            # Potentially add attention slicing/xformers/etc. for memory efficiency
            # enable_attention_slicing="auto", # Can help with VRAM
            # enable_xformers_memory_efficient_attention() # Needs xformers installed and a compatible GPU
        )
        end_time = time.time()
        print(f"Generation finished in {end_time - start_time:.2f} seconds.")
        # output.images will be a list of PIL Images
        generated_images_list = output.images

        # Determine the seed to return: the one we attempted to use, or -1 if generator creation failed
        # Note: When num_images_per_prompt > 1, all images use the same *initial* seed/generator state,
        # but the underlying noise is typically different unless a specific batching mechanism is used.
        # Reporting the main seed_int is standard practice.
        actual_seed_used = seed_int # Return the seed we used or attempted to use

        # Return the list of images and the seed
        return generated_images_list, actual_seed_used

    except gr.Error as e:
         # Re-raise Gradio errors directly
         raise e
    except ValueError as ve:
         # Handle specific value errors like invalid parameters
         print(f"Parameter Error: {ve}")
         raise gr.Error(f"Invalid Parameter: {ve}")
    except Exception as e:
        # Catch any other unexpected errors during generation
        print(f"An error occurred during image generation: {e}")
        error_message_lower = str(e).lower()
        if "size must be a multiple of" in error_message_lower or "invalid dimensions" in error_message_lower or "shape mismatch" in error_message_lower:
             raise gr.Error(f"Image generation failed - Invalid size '{width}x{height}' for model: {e}. Try a multiple of 64 or 8.")
        elif "out of memory" in error_message_lower or "cuda out of memory" in error_message_lower:
             print("Hint: Try smaller image size, fewer steps, fewer images, or a model that uses less VRAM.") # Added "fewer images" hint
             raise gr.Error(f"Out of Memory (OOM) during generation. Try smaller size/steps, fewer images, or select CPU. Error: {e}") # Added "fewer images" hint
        elif "runtimeerror" in error_message_lower:
             raise gr.Error(f"Runtime Error during generation: {e}. This could be a model/scheduler incompatibility or other issue.")
        elif "device-side assert" in error_message_lower or "cuda error" in error_message_lower:
             raise gr.Error(f"CUDA/GPU Error during generation: {e}. Ensure PyTorch with CUDA is correctly installed and compatible.")
        elif "expected all tensors to be on the same device" in error_message_lower:
             raise gr.Error(f"Device mismatch error during generation: {e}. This is an internal error, please report it.")
        else:
             # Generic catch-all for unknown errors
             raise gr.Error(f"Image generation failed: An unexpected error occurred. {e}")

# --- Gradio Interface ---
local_models = list_local_models(MODELS_DIR)
# Combine the Style names from the map with the local model paths (prepended for clarity)
styled_models = list(STYLE_MODEL_MAP.keys())
additional_local_model_names = [f"Local: {path}" for path in local_models]

# Combine styles and local models for the dropdown choices
model_choices = styled_models + additional_local_model_names


if not model_choices:
    initial_model_choices = ["No models found"]
    initial_default_model = "No models found"
    model_dropdown_interactive = False
    print(f"\n--- IMPORTANT ---")
    print(f"No models available!")
    print(f"Please define Styles and their corresponding Hub models in STYLE_MODEL_MAP in main.py")
    print(f"or place additional local diffusers models in '{os.path.abspath(MODELS_DIR)}'.")
    print(f"-----------------\n")
else:
    initial_model_choices = model_choices
    # Set a reasonable default: prioritize the first style if available, otherwise the first local model
    if styled_models:
         initial_default_model = styled_models[0] # Default to the first defined style
    elif additional_local_model_names:
         initial_default_model = additional_local_model_names[0] # Default to the first local if no styles
    else: # Should not happen if model_choices is not empty
         initial_default_model = model_choices[0] # Fallback to first item

    model_dropdown_interactive = True

scheduler_choices = list(SCHEDULER_MAP.keys())

with gr.Blocks(theme=gr.themes.Soft()) as demo: # Added a soft theme for better aesthetics
    gr.Markdown(
        f"""
        # Perchance Revival
        Recreating the experience of the old Perchance Stable Diffusion generator.
        Select a style below, enter your prompt, and generate!
        Note: this app is currently in early development - UI will be adjusted to match Perchance presets.
        Note: 'hire.fix' size option currently generates at 1024x1024.
        Have fun!
        """
    )

    with gr.Row():
        with gr.Column(scale=2): # Give more space to controls
            # Updated label to reflect 'Styles' and local models
            model_dropdown = gr.Dropdown(
                choices=initial_model_choices,
                value=initial_default_model,
                label=f"Select Style / Model (Featured Styles or Additional Local from ./{MODELS_DIR})",
                interactive=model_dropdown_interactive,
            )
            device_dropdown = gr.Dropdown(
                choices=AVAILABLE_DEVICES,
                value=DEFAULT_DEVICE,
                label="Processing Device",
                interactive=len(AVAILABLE_DEVICES) > 1, # Only make interactive if both CPU and GPU are options
            )
            prompt_input = gr.Textbox(label="Positive Prompt", placeholder="Enter your prompt...", lines=3, autofocus=True) # Autofocus on prompt - generic placeholder
            negative_prompt_input = gr.Textbox(label="Negative Prompt (Optional)", placeholder="Enter negative prompt (e.g. blurry, bad quality)...", lines=2) # generic placeholder

            with gr.Accordion("Advanced Settings", open=False): # Keep advanced settings initially closed
                with gr.Row():
                    steps_slider = gr.Slider(minimum=5, maximum=150, value=30, label="Inference Steps", step=1)
                    cfg_slider = gr.Slider(minimum=1.0, maximum=30.0, value=7.5, label="CFG Scale", step=0.1) # Increased max CFG
                with gr.Row():
                     scheduler_dropdown = gr.Dropdown(
                        choices=scheduler_choices,
                        value=DEFAULT_SCHEDULER,
                        label="Scheduler"
                    )
                     size_dropdown = gr.Dropdown(
                        choices=SUPPORTED_SD15_SIZES,
                        value="512x512",
                        label="Image Size"
                    )
                with gr.Row(): # Group Seed and Images together
                     seed_input = gr.Number(label="Seed (-1 for random)", value=-1, precision=0) # precision=0 for integer
                     num_images_slider = gr.Slider(
                         minimum=1,
                         maximum=4, # Set a reasonable max for typical hardware/VRAM
                         value=1,
                         step=1,
                         label="Number of Images",
                         interactive=True
                     )


            generate_button = gr.Button("✨ Generate Image ✨", variant="primary", scale=1)

        # Change output from gr.Image to gr.Gallery
        with gr.Column(scale=3): # Give more space to output
            # Changed from gr.Image
            output_gallery = gr.Gallery( # Changed component type
                label="Generated Images", # Changed label
                show_label=True, # Ensure label is shown
                show_share_button=True,
                show_download_button=True,
                interactive=False # Output is not interactive
            )
             # Add a display for the actual seed used
            actual_seed_output = gr.Number(label="Actual Seed Used", precision=0, interactive=False)


    # Link button click to generation function
    generate_button.click(
        fn=generate_image,
        inputs=[
            model_dropdown, # This now passes the Style Name or "Local: path"
            device_dropdown,
            prompt_input,
            negative_prompt_input,
            steps_slider,
            cfg_slider,
            scheduler_dropdown,
            size_dropdown,
            seed_input,
            num_images_slider
        ],
        outputs=[output_gallery, actual_seed_output],
        api_name="generate" # Optional: For API access
    )

    # Add some notes/footer
    gr.Markdown(
        f"""
        ---
        **Usage Notes:**

        1. The featured "Styles" are defined in `main.py` and map to specific Stable Diffusion 1.5 models hosted on Hugging Face.
        2. You can add *additional* local Diffusers-compatible SD 1.5 models into the `./{MODELS_DIR}` folder; they will appear in the dropdown prefixed with "Local:".
        3. Select a Style/Model from the dropdown.
        4. Choose your processing device (GPU recommended if available).
        5. Enter your positive and optional negative prompts.
        6. Optional: Adjust advanced settings (Steps, CFG Scale, Scheduler, Size, Seed, Number of Images).
        7. Click "Generate Image".
        8. Have fun!
        The first generation with a new Style/Model might take some time to load as the model is initially downloaded from the hub to your local storage.
        Generating multiple images increases VRAM and time requirements.
        """
    )


if __name__ == "__main__":
    print("\n--- Starting Perchance Revival ---")
    cuda_status = "CUDA available" if torch.cuda.is_available() else "CUDA not available"
    gpu_count_str = f"Found {torch.cuda.device_count()} GPU(s)." if torch.cuda.is_available() else ""

    print(f"{cuda_status} {gpu_count_str}")
    print(f"Available devices detected by PyTorch: {', '.join(AVAILABLE_DEVICES)}")
    print(f"Default device selected by app: {DEFAULT_DEVICE}")

    if not model_choices:
         print(f"\n!!! WARNING: No models available. The Gradio app will launch but cannot generate images.")
         print(f"Please define Styles and their corresponding Hub models in STYLE_MODEL_MAP in main.py")
         print(f"or add additional local diffusers models to '{MODELS_DIR}'. !!!")
    else:
         num_styled = len(STYLE_MODEL_MAP)
         num_additional_local = len(local_models)
         print(f"Defined {num_styled} featured style(s) mapping to Hub models.")
         print(f"Found {num_additional_local} additional local model(s) in '{os.path.abspath(MODELS_DIR)}'.")


    # Optional: Hugging Face login if needed for gated models (requires uncommenting imports and adding login logic)
    # print("Checking Hugging Face login status...")
    # try:
    #      token = HfFolder.get_token()
    #      if token is None:
    #           print("Hugging Face token not found. You might need to log in to Hugging Face for some Hub models.")
    #           print("Run `huggingface-cli login` in your terminal if needed.")
    #      else:
    #          print("Hugging Face token found.")
    # except Exception as e:
    #     print(f"Could not check Hugging Face token: {e}")


    print("Launching Gradio interface...")
    # Use share=True if you want to share a public link (e.g., for testing or demo)
    # auth=('username', 'password') can be added for basic authentication if sharing
    # server_name="0.0.0.0" if you want to access from other machines on the network
    # server_port=7860 # Default port
    demo.launch(show_error=True, inbrowser=True) # Launch in browser by default
    print("Gradio interface closed.")
