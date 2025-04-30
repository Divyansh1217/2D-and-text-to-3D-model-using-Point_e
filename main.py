
import torch
from tqdm.auto import tqdm
import clip
from PIL import Image
import streamlit as st
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.util.pc_to_mesh import marching_cubes_mesh
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud
from point_e.util.point_cloud import PointCloud

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('creating base model...')
base_name = 'base40M-textvec'
base_model = model_from_config(MODEL_CONFIGS[base_name], device)
base_model.eval()
base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

print('creating upsample model...')
upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
upsampler_model.eval()
upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

base_ckpt_path = 'point_e_model_cache/base_40m_textvec.pt'
upsample_ckpt_path = 'point_e_model_cache/upsample_40m.pt'

print('loading base checkpoint from local...')
base_model.load_state_dict(torch.load(base_ckpt_path, map_location=device))

print('loading upsampler checkpoint from local...')
upsampler_model.load_state_dict(torch.load(upsample_ckpt_path, map_location=device))
sampler = PointCloudSampler(
    device=device,
    models=[base_model, upsampler_model],
    diffusions=[base_diffusion, upsampler_diffusion],
    num_points=[1024, 4096 - 1024],
    aux_channels=['R', 'G', 'B'],
    guidance_scale=[3.0, 3.0],
    model_kwargs_key_filter=('texts', ''), # Do not condition the upsampler at all
)
# Set a prompt to condition on.
see=st.radio("Select the prompt: ", ("Text", "Image"))
if see=="Text":
        prompt = st.text_input("Write the prompt")
        samples = None
        for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt]))):
            samples = x
        pc = sampler.output_to_point_clouds(samples)[0]
        fig = plot_point_cloud(pc, grid_size=3, fixed_bounds=((-0.75, -0.75, -0.75),(0.75, 0.75, 0.75)))
else:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            prompt = Image.open(uploaded_file)
            @st.cache_resource
            def load_clip_model():
                model, preprocess = clip.load("point_e_model_cache/ViT-L-14.pt", device=device, download_root="clip_model_cache")
                return model, preprocess
            def get_image_embedding(image):
                model, preprocess = load_clip_model()
                image_input = preprocess(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_features = model.encode_image(image_input)
                print("image is converted into embeddings")
                return image_features
            image_embeddings=get_image_embedding(prompt)


            st.image(prompt, caption="Uploaded Image", use_column_width=True)   
            samples = None
            for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(images=image_embeddings))):
                samples = x
            pc = sampler.output_to_point_clouds(samples)[0]
            fig = plot_point_cloud(pc, grid_size=3, fixed_bounds=((-0.75, -0.75, -0.75),(0.75, 0.75, 0.75)))
            pc.save("exg.npz")


print('creating SDF model...')
name = 'sdf'
model = model_from_config(MODEL_CONFIGS[name], device)
model.eval()

print('loading SDF model...')
model.load_state_dict(load_checkpoint(name, device))
# Produce a sample from the model.
mesh = marching_cubes_mesh(
    pc=pc,
    model=model,
    batch_size=4096,
    grid_size=32, # increase to 128 for resolution used in evals
    progress=True,
)

# Write the mesh to a PLY file to import into some other program.
with open('mesh.ply', 'wb') as f:
    mesh.write_ply(f)



    

