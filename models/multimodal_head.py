import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import numpy as np
from PIL import Image

class MultimodalHead(nn.Module):
    def __init__(self,
                 dim: int,
                 image_size: int = 256,
                 patch_size: int = 16,
                 num_mesh_vertices: int = 1024):
        super().__init__()
        self.dim = dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_mesh_vertices = num_mesh_vertices
        
        # Text generation (byte-level)
        self.to_bytes = nn.Linear(dim, 256)
        
        # Image generation with improved architecture
        self.image_head = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.LayerNorm(dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, patch_size * patch_size * 3),
        )
        
        # Mesh generation components
        self.vertex_generator = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, num_mesh_vertices * 3)  # xyz coordinates
        )
        
        self.face_generator = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, (num_mesh_vertices - 2) * 3)  # triangles
        )
        
        # Rotation predictor
        self.rotation_predictor = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 9)  # 3x3 rotation matrix
        )
        
    def generate_text(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        logits = self.to_bytes(x) / temperature
        return F.softmax(logits, dim=-1)
        
    def generate_image(self, x: torch.Tensor) -> torch.Tensor:
        # Get batch size and sequence length
        batch_size = x.size(0)
        
        # Generate image features
        features = self.image_head(x)  # [B, seq_len, patch_size * patch_size * 3]
        
        # Reshape to proper image format
        features = features.view(batch_size, -1, 3, self.patch_size, self.patch_size)
        features = features.mean(dim=1)  # Average over sequence dimension
        
        # Ensure we have [B, C, H, W] format
        if len(features.shape) != 4:
            raise ValueError(f"Expected 4D tensor [B,C,H,W], got shape {features.shape}")
        
        # Upsample to full resolution
        image = F.interpolate(
            features, 
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False
        )
        
        return torch.sigmoid(image)
        
    def generate_mesh(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate mesh directly from features"""
        # Generate vertices
        vertices = self.vertex_generator(x)
        vertices = vertices.view(-1, self.num_mesh_vertices, 3)
        
        # Generate faces (triangles)
        faces = self.face_generator(x)
        faces = faces.view(-1, self.num_mesh_vertices - 2, 3)
        faces = F.softmax(faces, dim=-1)  # Convert to probabilities
        
        return {
            'vertices': vertices,
            'faces': faces
        }
        
    def rotate_mesh(self, vertices: torch.Tensor, rotation_matrix: torch.Tensor) -> torch.Tensor:
        """Apply rotation to mesh vertices"""
        # Ensure rotation matrix is orthogonal
        u, _, v = torch.svd(rotation_matrix)
        rotation_matrix = torch.matmul(u, v.transpose(-2, -1))
        
        # Apply rotation
        rotated_vertices = torch.matmul(vertices, rotation_matrix.T)
        return rotated_vertices
        
    def predict_rotation(self, x: torch.Tensor) -> torch.Tensor:
        """Predict rotation matrix from features"""
        rotation = self.rotation_predictor(x)
        rotation = rotation.view(-1, 3, 3)
        
        # Ensure rotation matrix is orthogonal
        u, _, v = torch.svd(rotation)
        rotation = torch.matmul(u, v.transpose(-2, -1))
        
        return rotation
        
    def render_mesh(self, vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        """Simple mesh rendering to image"""
        # Project 3D vertices to 2D
        # Simple orthographic projection for now
        projected = vertices[..., :2]  # Take x,y coordinates
        
        # Scale to image size
        projected = (projected + 1) * (self.image_size / 2)
        
        # Create empty image
        image = torch.zeros(3, self.image_size, self.image_size)
        
        # Simple rendering - just show vertex positions
        # In practice, you'd want proper rasterization
        for vertex in projected:
            x, y = vertex.long()
            if 0 <= x < self.image_size and 0 <= y < self.image_size:
                image[:, y, x] = 1.0
                
        return image
        
    def forward(self,
                x: torch.Tensor,
                output_type: str = 'text',
                temperature: float = 1.0,
                spatial_task: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        
        outputs = {}
        
        if spatial_task is not None:
            if spatial_task['type'] == 'rotate':
                # Generate mesh
                mesh = self.generate_mesh(x)
                
                if 'rotation_angles' in spatial_task:
                    # Use provided rotation
                    rotation = spatial_task['rotation_angles'].view(3, 3)
                else:
                    # Predict rotation
                    rotation = self.predict_rotation(x)
                
                # Rotate mesh
                rotated_vertices = self.rotate_mesh(mesh['vertices'], rotation)
                
                # Render views
                original_image = self.render_mesh(mesh['vertices'], mesh['faces'])
                rotated_image = self.render_mesh(rotated_vertices, mesh['faces'])
                
                outputs.update({
                    'original_mesh': mesh,
                    'rotated_vertices': rotated_vertices,
                    'original_image': original_image,
                    'rotated_image': rotated_image,
                    'rotation_matrix': rotation
                })
                
        # Standard output generation
        if output_type == 'text' or output_type == 'multimodal':
            # Generate text (byte-level)
            logits = self.to_bytes(x) / temperature
            outputs['text'] = logits
            
        if output_type == 'image' or output_type == 'multimodal':
            # Generate image
            outputs['image'] = self.generate_image(x)
            
        if output_type == 'mesh' or output_type == 'multimodal':
            # Generate mesh
            vertices = self.vertex_generator(x).view(-1, self.num_mesh_vertices, 3)
            faces = self.face_generator(x).view(-1, self.num_mesh_vertices-2, 3)
            outputs['mesh'] = {
                'vertices': vertices,
                'faces': faces
            }
            
        return outputs
