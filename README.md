Project Structure:

project/
├── models/
│   ├── __init__.py
│   ├── transformer_squared.py
│   ├── byte_latent_patches.py
│   ├── diff_attention.py
|   └── multimodal_head.py
|   └── MDM.py
|
├── config/
│   ├── __init__.py
│   ├── model_config.py
│   └── model_config.json
├── train.py
└── chat.py



I want it to be a multimodal model that can take text, images,video,audio 3d meshes as inputs
with the byte level and output text, images and 3d meshes. With a self adaptive transformer structure
and surprise based memory as a layer. With chains of thought that can use image and 3d meshes as context.


