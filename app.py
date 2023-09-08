import streamlit as st
import glob
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import torch

img_names = list(glob.glob('pics/*.jpg'))[:2000]

@st.cache(allow_output_mutation=True)

def load_model():
    model = SentenceTransformer('clip-ViT-B-32') # clip-ViT-B-32 for eng <= only
    loaded_embeddings =model.encode([Image.open(img) for img in img_names], batch_size=32, convert_to_tensor=True, show_progress_bar=True)
    
    return model, loaded_embeddings

def search(query, k=3):
    # Clear the images list before appending new results
    images.clear()
    
    query_emd = model.encode([query], convert_to_tensor=True, show_progress_bar=False)
    hits = util.semantic_search(query_emd, loaded_embeddings, top_k=k)[0]

    for hit in hits:
        img_path = img_names[hit['corpus_id']]
        images.append(img_path)

# Streamlit UI
st.title("Contextual Image Viewer App")

# Load the machine learning model
model, loaded_embeddings = load_model()

# Initialize the images list
images = []

# Choose input type
input_type = st.radio("Select input type:", ("Text", "Image"))

if input_type == "Text":
    text_input = st.text_input("Enter text:")
    if st.button("Submit"):
        search(text_input)
        
        # Display the relevant images
        for image_path in images:
            st.image(image_path, caption=image_path, use_column_width=True)

else:  # Input type is Image
    uploaded_image = st.file_uploader("Upload an image:", type=["jpg", "png", "jpeg"])
    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        
        # Call your function to get relevant images based on the uploaded image (you may need to modify this)
        img = Image.open(uploaded_image)
        search(img)
        
        # Display the relevant images
        for image_path in images:
            st.image(image_path, caption=image_path, use_column_width=True)
