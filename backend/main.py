import modal

image = (
    modal.Image.debian_slim()
    .run_commands(
        "apt update && apt install -y git portaudio19-dev",
        "git clone https://github.com/sd-fabric/fabric.git")
    ).pip_install_from_requirements("requirements.txt")
    

stub = modal.Stub("eeg-art", image=image)

generator = None
images = []

if not modal.is_local():
    from fabric.generator import AttentionBasedGenerator
    from fabric.iterative import IterativeFeedbackGenerator
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    from PIL import Image
    model_name = "dreamlike-art/dreamlike-photoreal-2.0"  # @param {type:"string"}
    model_ckpt = ""  # @param {type:"string"}

    model_name = model_name if model_name else None
    model_ckpt = model_ckpt if model_ckpt else None

    generator = AttentionBasedGenerator(
        model_name=model_name,
        model_ckpt=model_ckpt,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    if torch.cuda.is_available():
        generator.to("cuda")

    generator = IterativeFeedbackGenerator(generator)

@stub.function(keep_warm=1, concurrency_limit=1, gpu=modal.gpu.A100(memory=20))
@modal.web_endpoint(method="GET")
def root(prompt: str = "photo of a dog running on grassland, masterpiece, best quality, fine details", negative_prompt: str = "lowres, bad anatomy, bad hands, cropped, worst quality",
                   denoising_steps = 20,
                   guidance_scale = 6.0,
                   feedback_start = 0.0,
                   feedback_end = 0.5,
                   img_index: int = None):
    
    if img_index is not None:
        generator.give_feedback(liked=images[img_index])
        return images[img_index]
    
    images = generator.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        denoising_steps=denoising_steps,
        guidance_scale=guidance_scale,
        feedback_start=feedback_start,
        feedback_end=feedback_end,
    )

    return images




# @markdown # Running FABRIC

# @markdown Explanation of the parameters:
# @markdown - `denoising_steps`: Number of steps in the denoising schedule
# @markdown - `guidance_scale`: Strength of the classifier-free guidance (same as for any diffusion model)
# @markdown - `feedback_start`: From which point in the diffusion process feedback should be added (0.0 -> from the beginning, 0.5 -> from the halfway point)
# @markdown - `feedback_end`: Until which point feedback should be added (0.5 -> until the halfway point, 1.0 -> until the end)
# @markdown
# @markdown **NOTE**: GPU memory scales with the number of feedback images, so a large number of feedback images will require large amounts of memory.


# def display_images(images, n_cols=4, size=4):
#     n_rows = int(np.ceil(len(images) / n_cols))
#     fig = plt.figure(figsize=(size * n_cols, size * n_rows))
#     for i, img in enumerate(images):
#         ax = fig.add_subplot(n_rows, n_cols, i + 1)
#         ax.imshow(img)
#         ax.set_title(f"Image {i+1}")
#         ax.axis("off")
#     fig.tight_layout()
#     return fig


# def get_selected_idx(prompt, min_idx=0, max_idx=3):
#     range_str = ", ".join(map(str, range(min_idx + 1, max_idx + 2)))
#     selected_idx = input(f"{prompt} ({range_str}) ")

#     if not selected_idx:
#         return None

#     try:
#         selected_idx = int(selected_idx)
#     except:
#         print(f"Unable to parse '{selected_idx}', selecting no feedback.")
#         return None
#     else:
#         if selected_idx < min_idx + 1 or selected_idx > max_idx + 1:
#             print("Index out of bounds, selecting no feedback.")
#             return None
#         else:
#             return selected_idx - 1


# def get_feedback(images) -> tuple[list[Image.Image], list[Image.Image]]:
#     display_images(images)
#     plt.show()
#     # liked_idx = get_selected_idx("Which image do you like most?")
#     # disliked_idx = get_selected_idx("Which image do you like least?")


# import ipywidgets as widgets
# import functools
# from IPython.display import display

# images = []


# def clicked_like(img, i, _):
#     generator.give_feedback(liked=[img])
#     text = widgets.Label(value=f"Added image {i+1} to liked images")
#     display(text)


# def clicked_dislike(img, i, _):
#     generator.give_feedback(disliked=[img])
#     text = widgets.Label(value=f"Added image {i+1} to disliked images")
#     display(text)


# like_buttons = []
# dislike_buttons = []
# for i in range(4):
#     like_button = widgets.Button(
#         description=f"👍 Image {i+1}",
#         button_style="success",
#         tooltip="Add to liked images",
#     )
#     like_buttons.append(like_button)

#     dislike_button = widgets.Button(
#         description=f"👎 Image {i+1}",
#         button_style="danger",
#         tooltip="Add to disliked images",
#     )
#     dislike_buttons.append(dislike_button)

# like_container = widgets.HBox(like_buttons)
# dislike_container = widgets.HBox(dislike_buttons)


# def next_round(_):
#     clear_output()
#     images = generator.generate(
#         prompt=prompt,
#         negative_prompt=negative_prompt,
#         denoising_steps=denoising_steps,
#         guidance_scale=guidance_scale,
#         feedback_start=feedback_start,
#         feedback_end=feedback_end,
#     )
#     clear_output()

#     display_images(images)
#     plt.show()

#     for i in range(4):
#         like_buttons[i]._click_handlers.callbacks = []
#         dislike_buttons[i]._click_handlers.callbacks = []
#         like_buttons[i].on_click(functools.partial(clicked_like, images[i], i))
#         dislike_buttons[i].on_click(functools.partial(clicked_dislike, images[i], i))

#     display(like_container)
#     display(dislike_container)
#     display(control_buttons)


# def reset(_):
#     generator.reset()
#     text = widgets.Label(value="All feedback images have been cleared.")
#     display(text)


# next_round_button = widgets.Button(description="Next Round", button_style="info")
# next_round_button.on_click(next_round)
# reset_button = widgets.Button(
#     description="Reset Feedback", tooltip="Clear all feedback images"
# )
# reset_button.on_click(reset)
# control_buttons = widgets.HBox([next_round_button, reset_button])

# generator.reset()
# next_round(None)
