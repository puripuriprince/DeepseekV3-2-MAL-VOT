Sushi is a multimodal model that can take text, images, video, audio, 3d meshes as inputs
at the byte level and output text, images and 3d meshes. With a self adaptive transformer^2 structure and surprise based memory as a layer using titans_pytorch. With chains of thought that can use image and 3d meshes as context.


This model is trained by DeepseekV3 using model distillation. And then finetuned on the Sushi dataset.

Project Structure:

Byte_latent_patches.py contains the BLT.txt paper implementation.

Diff_attention.py contains the differential attention paper implementation.

Transformer_squared.py contains the transformer^2 paper implementation.

SushiFull.py contains the titans paper implementation through the titans_pytorch library.

Thinking.py contains the chains of thought implementation inlcuding images in thought from the  imagesincahinofthought.txt paper.







