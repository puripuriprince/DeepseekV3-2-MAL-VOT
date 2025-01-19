import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image
import trimesh
from einops import rearrange, repeat

class UniversalProcessor(nn.Module):
    def __init__(self,
                 dim: int,
                 max_sequence_length: int = 8192,
                 image_size: int = 256,
                 patch_size: int = 16,
                 mesh_max_vertices: int = 1024,
                 use_deepseek: bool = True):
        super().__init__()
        self.dim = dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.mesh_max_vertices = mesh_max_vertices
        
        # DeepSeek base embeddings
        #if use_deepseek:
            #from deepseek import DeepSeekEmbeddings
           # self.base_embeddings = #DeepSeekEmbeddings(dim)
        #else:
        self.byte_embeddings = nn.Embedding(256, dim)
            
        # Learnable position embeddings with interpolation
        self.pos_embedding = nn.Parameter(torch.randn(1, max_sequence_length, dim))
        
        # Dynamic image processing with integer keys
        self.patch_embeddings = nn.ModuleDict({
            '256': nn.Sequential(
                nn.Conv2d(3, dim, kernel_size=16, stride=16),
                nn.LayerNorm([dim, 16, 16])
            ),
            '384': nn.Sequential(
                nn.Conv2d(3, dim, kernel_size=24, stride=24),
                nn.LayerNorm([dim, 16, 16])
            ),
            '512': nn.Sequential(
                nn.Conv2d(3, dim, kernel_size=32, stride=32),
                nn.LayerNorm([dim, 16, 16])
            )
        })
        self.supported_sizes = [256, 384, 512]  # Store sizes as integers
        
        # Enhanced mesh embeddings
        self.vertex_embeddings = nn.Sequential(
            nn.Linear(3, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim)
        )
        self.face_embeddings = nn.Sequential(
            nn.Linear(9, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim)
        )
        
        # Video frame processing
        self.temporal_conv = nn.Conv3d(dim, dim, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.temporal_pos = nn.Parameter(torch.randn(1, 32, 1, dim))  # Up to 32 frames
        
        # Type embeddings
        self.type_embeddings = nn.Embedding(6, dim)  # text, image, mesh, video, audio, mixed
        
    def interpolate_pos_encoding(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Interpolate position encodings for variable sequence lengths"""
        pos_emb = self.pos_embedding
        if seq_len != pos_emb.shape[1]:
            pos = torch.linspace(0, pos_emb.shape[1]-1, seq_len, device=pos_emb.device)
            pos_emb = F.interpolate(
                pos_emb.transpose(1, 2),
                size=seq_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        return pos_emb[:, :seq_len]
        
    def process_bytes(self, text: Union[str, bytes, torch.Tensor]) -> torch.Tensor:
        if hasattr(self, 'base_embeddings'):
            return self.base_embeddings(text)
        
        # If input is already a tensor, use it directly
        if isinstance(text, torch.Tensor):
            if text.dim() == 1:
                text = text.unsqueeze(0)  # Add batch dimension if needed
            return self.byte_embeddings(text)
        
        # Handle string or bytes input
        if isinstance(text, str):
            bytes_data = text.encode('utf-8')
        else:
            bytes_data = text
            
        # Convert bytes to tensor with proper shape [batch_size, sequence_length]
        tokens = torch.tensor([list(bytes_data)], dtype=torch.long, device=self.byte_embeddings.weight.device)
        embeddings = self.byte_embeddings(tokens)
        return embeddings
        
    def process_image(self, image: Union[str, Image.Image, torch.Tensor]) -> torch.Tensor:
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        if isinstance(image, Image.Image):
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            
        # Select appropriate patch embedding based on image size
        h, w = image.shape[-2:]
        target_size = max(h, w)
        closest_size = min(self.supported_sizes, key=lambda x: abs(x - target_size))
        
        # Resize if needed
        if (h, w) != (closest_size, closest_size):
            image = F.interpolate(image.unsqueeze(0), size=(closest_size, closest_size), 
                                mode='bilinear', align_corners=False).squeeze(0)
            
        # Get patches using string key
        patches = self.patch_embeddings[str(closest_size)](image)
        return patches.flatten(2).transpose(1, 2)
        
    def process_video(self, video: torch.Tensor) -> torch.Tensor:
        """Process video frames with temporal context"""
        # video shape: [batch, frames, channels, height, width]
        b, f, c, h, w = video.shape
        
        # Process each frame
        frame_embeddings = []
        for i in range(f):
            frame_emb = self.process_image(video[:, i])
            frame_embeddings.append(frame_emb)
            
        # Stack and apply temporal convolution
        x = torch.stack(frame_embeddings, dim=1)  # [batch, frames, patches, dim]
        x = rearrange(x, 'b f p d -> b d f p')
        x = self.temporal_conv(x)
        x = rearrange(x, 'b d f p -> b f p d')
        
        # Add temporal position embeddings
        x = x + self.temporal_pos[:, :f]
        
        return x.reshape(b, f * x.size(2), self.dim)
        
    def process_mesh(self, mesh_input: Union[str, trimesh.Trimesh, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        if isinstance(mesh_input, tuple):
            # Handle direct tensor input (vertices, faces)
            vertices, faces = mesh_input
            
            # Process vertices
            vertex_embeds = self.vertex_embeddings(vertices)  # [batch, num_vertices, dim]
            
            # Process faces - reshape and expand to match 9D input
            faces_flat = faces.float().reshape(-1, 3)  # [batch * num_faces, 3]
            faces_expanded = torch.cat([
                faces_flat,
                faces_flat,  # Repeat the indices to create 9D vectors
                faces_flat
            ], dim=-1)  # [batch * num_faces, 9]
            
            face_embeds = self.face_embeddings(faces_expanded)  # [batch * num_faces, dim]
            
            # Reshape face embeddings to match batch dimension
            batch_size = vertices.size(0)
            num_faces = faces.size(1)
            face_embeds = face_embeds.view(batch_size, num_faces, -1)  # [batch, num_faces, dim]
            
            # Combine embeddings along sequence dimension
            combined = torch.cat([vertex_embeds, face_embeds], dim=1)  # [batch, num_vertices + num_faces, dim]
            
            return combined
        
        elif isinstance(mesh_input, str):
            mesh = trimesh.load(mesh_input)
        elif isinstance(mesh_input, trimesh.Trimesh):
            mesh = mesh_input
        else:
            raise ValueError(f"Unsupported mesh input type: {type(mesh_input)}")
            
        # Process trimesh object
        vertices = torch.from_numpy(mesh.vertices[:self.mesh_max_vertices]).float()
        vertex_normals = torch.from_numpy(mesh.vertex_normals[:self.mesh_max_vertices]).float()
        
        # Combine position and normal information
        vertex_features = torch.cat([vertices, vertex_normals], dim=-1)
        vertex_embeds = self.vertex_embeddings(vertex_features)
        
        # Process faces with topology awareness
        faces = torch.from_numpy(mesh.faces[:self.mesh_max_vertices]).float()
        face_embeds = self.face_embeddings(faces.reshape(-1, 9))
        
        # Combine with attention
        combined = torch.cat([vertex_embeds, face_embeds], dim=0)
        
        return combined
        
    def forward(self, inputs: Union[str, torch.Tensor, Dict[str, Any]], input_type: Optional[str] = None) -> torch.Tensor:
        if isinstance(inputs, dict):
            # Handle multimodal input
            embeddings = []
            for key, value in inputs.items():
                if key == 'text':
                    emb = self.process_bytes(value)
                elif key == 'image':
                    emb = self.process_image(value)
                elif key == 'mesh':
                    emb = self.process_mesh(value)
                elif key == 'video':
                    emb = self.process_video(value)
                embeddings.append(emb)
            x = torch.cat(embeddings, dim=1)
            type_id = 5  # mixed
        else:
            # Single modality input
            if input_type == 'text' or input_type == 'bytes' or input_type is None:
                x = self.process_bytes(inputs)
                type_id = 0
            elif input_type == 'image':
                x = self.process_image(inputs)
                type_id = 1
            elif input_type == 'mesh':
                x = self.process_mesh(inputs)
                type_id = 2
            elif input_type == 'video':
                x = self.process_video(inputs)
                type_id = 3
                
        # Add type embeddings
        type_emb = self.type_embeddings(torch.tensor([type_id], device=x.device))
        x = x + type_emb.unsqueeze(1)  # Add sequence dimension to type embedding
        
        # Add interpolated position embeddings
        x = x + self.interpolate_pos_encoding(x, x.size(1))
        
        return x
