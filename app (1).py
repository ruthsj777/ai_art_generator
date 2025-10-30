import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

# Title & description
st.title("🎨 AI Art Generator")
st.markdown("Type a creative prompt and let AI paint it for you!")

# Load Stable Diffusion model
@st.cache_resource
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

pipe = load_model()

# User input
prompt = st.text_input("Enter your art prompt:",
                       "A dreamy castle floating on clouds in watercolor style")

# Generate image
if st.button("Generate Art"):
    with st.spinner("🎨 Creating your masterpiece... please wait ⏳"):
        image = pipe(prompt).images[0]
        st.image(image, caption="Generated Art", use_container_width=True)
