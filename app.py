# app.py
import streamlit as st
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from model import UNet, NoiseScheduler, Config

cfg = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_model():
    model = UNet(cfg.CHANNELS, cfg.BASE_CHANNELS, cfg.TIME_EMB_DIM).to(device)
    ck = torch.load('best.pt', map_location=device)
    model.load_state_dict(ck['model_state'])
    model.eval()
    scheduler = NoiseScheduler(cfg.T, cfg.BETA_START, cfg.BETA_END,
                                schedule=cfg.NOISE_SCHEDULE, device=device)
    return model, scheduler

def denorm(t):
    return (t.clamp(-1, 1) + 1) / 2

def to_pil(t):
    arr = (denorm(t).permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)

st.title("🎨 DDPM — Face Generator")
st.markdown("Denoising Diffusion Probabilistic Model trained on CelebA-HQ")

model, scheduler = load_model()

tab1, tab2 = st.tabs(["Generate from Noise", "Reconstruct Image"])

with tab1:
    n_imgs = st.slider("Number of images", 1, 5, 3)
    if st.button("Generate!", type="primary"):
        with st.spinner("Generating..."):
            with torch.no_grad():
                imgs = scheduler.p_sample_loop(
                    model, (n_imgs, cfg.CHANNELS, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))
            cols = st.columns(n_imgs)
            for i, col in enumerate(cols):
                col.image(to_pil(imgs[i]), caption=f"Generated #{i+1}")

with tab2:
    uploaded = st.file_uploader("Upload a face image", type=["jpg", "png", "jpeg"])
    noise_t = st.slider("Noise strength", 50, 280, 200, step=10)
    if uploaded and st.button("Reconstruct!", type="primary"):
        img_pil = Image.open(uploaded).convert("RGB")
        tf = T.Compose([T.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
                        T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
        img_t = tf(img_pil).unsqueeze(0)
        with st.spinner("Reconstructing..."):
            # add noise
            tt = torch.tensor([noise_t], device=device, dtype=torch.long)
            noisy, _ = scheduler.q_sample(img_t.to(device), tt)
            # denoise
            x = noisy.clone()
            for tv in range(noise_t, -1, -1):
                x = scheduler.p_sample(model, x, tv)
            recon = x.cpu()
        col1, col2, col3 = st.columns(3)
        col1.image(img_pil.resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)), caption="Original")
        col2.image(to_pil(noisy[0].cpu()), caption="Noisy")
        col3.image(to_pil(recon[0]), caption="Reconstructed")
