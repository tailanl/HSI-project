# Source boundary

This repository packages project-owned planning, geometry, pose conditioning,
refinement, memory, runtime adapters and tests. It does not redistribute scene
datasets, generated results, model weights, SMPL-X assets or external model
implementations.

Install the external implementations and obtain their assets separately:

- Qwen vision/text inference service: semantic planning and image review.
- ComfyUI with the required MiniMax H3 and SAM3 nodes: direct local model APIs.
- HybrIK-X and its detector: image-to-articulation recovery.
- SMPL-X and body segmentation/mean-hand assets: actual body geometry.
- ReMoGen, CLIP and their dependencies: the released denoiser, MVAE and text encoder.
- The polygon NavMesh Python package (`pynavmesh`): fresh navigation construction.

The project-owned denoiser adapter invokes the installed ReMoGen forward through
PyTorch hooks; it does not vendor that forward. Joint-driven skinning invokes
the installed SMPL-X numerical APIs. H3 uses the installed ComfyUI model API.
Third-party components and assets retain their respective terms; this source
package does not grant rights to redistribute them.

Some current artifact schema and diagnostic keys retain historical experiment
identifiers for data compatibility. They are not instructions to load historical
workspace code. Public configuration examples use placeholders only. Credentials
are supplied through environment variables and must not be committed.
