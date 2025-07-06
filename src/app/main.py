from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def read_root() -> dict:
    """Root endpoint returning a welcome message.

    Returns
    -------
    dict
        A dictionary with a welcome message.

    """
    return {"message": "Hello, World!"}
