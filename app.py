import gradio as gr
from app.main import app as fastapi_app

with gr.Blocks(title="Nectar Voice Agent") as demo:
    gr.HTML('<iframe src="/static/index.html" style="width:100%; height:100vh; border:none;"></iframe>')

app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=7860, reload=False)
