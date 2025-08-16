from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import ray
import uuid, json, redis, os

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
r = redis.from_url(REDIS_URL)
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return HTMLResponse(open("static/index.html").read())

@app.post("/create")
async def create_run(env: str = Form(...), algo: str = Form(...), max_iter: int = Form(2000)):
    run_id = str(uuid.uuid4())
    payload = dict(env=env, algo=algo, max_iter=max_iter, run_id=run_id)
    r.lpush("jobs", json.dumps(payload))
    return {"run_id": run_id}

@app.websocket("/ws/{run_id}")
async def stream(ws: WebSocket, run_id: str):
    await ws.accept()
    pubsub = r.pubsub()
    pubsub.subscribe(run_id)
    try:
        while True:
            msg = pubsub.get_message(ignore_subscribe_messages=True, timeout=1)
            if msg:
                await ws.send_text(msg["data"].decode())
    except WebSocketDisconnect:
        pass