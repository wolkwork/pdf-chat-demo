from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from langserve import add_routes


from .chains.simple_rag import chain  # type: ignore
from .chains.cited_rag import cited_chain  # type: ignore
from .utils import get_retriever


app = FastAPI()
templates = Jinja2Templates(directory="./src/templates")

retriever = get_retriever()


add_routes(app, chain, path="/simple", enable_feedback_endpoint=True)
add_routes(app, retriever, path="/semantic_search", enable_feedback_endpoint=True)


@app.get("/")
async def main(request: Request):
    context = {"request": request}
    return templates.TemplateResponse("main.html.jinja", context=context)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
