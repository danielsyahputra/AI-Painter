import streamlit as st
from utils.dataset import load_style_content
from utils.download import get_model
from utils.model import run_style_transfer

def settings() -> None:
    st.set_page_config(
        page_title="AI Painter",
        layout="centered",
    )
    global cnn
    cnn = get_model()
    # Title
    st.markdown("## AI Painter with Neural Style Transfer")

def main():
    settings()

    style_uploaded_file = st.file_uploader(label="Choose style image", type=['jpg', 'jpeg'])
    content_uploaded_file = st.file_uploader(label="Choose content image", type=['jpg', 'jpeg'])

    if not ((style_uploaded_file is None) or (content_uploaded_file is None)):
        style, content = load_style_content(content_uploaded_file=content_uploaded_file, style_uploaded_file=style_uploaded_file)
        input_img = content.clone()
        output = run_style_transfer(cnn=cnn, style=style, content=content, input_img=input_img)
        st.image(output)

if __name__=="__main__":
    main()