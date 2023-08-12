import asyncio
from websockets.server import serve
from io import BytesIO
from PIL import Image
from base64 import b64encode
import json


def b64enc(im):
    bio = BytesIO()
    im.resize((128, 128)).save(bio, format="JPEG")
    bio.seek(0)
    return b64encode(bio.read()).decode("utf-8")


def dummy(queue):
    async def fn(websocket):
        while True:
            im = Image.new("RGB", (512, 512))
            to_send = {"images": [b64enc(im)] * 4}
            await queue.put(to_send)
            await websocket.send(json.dumps(to_send))
            print("sent", str(to_send)[:50])
            avgs = json.loads(await websocket.recv())
            print(avgs)
    return fn

def retran(queue):
    async def fn(websocket):
        while True:
            data = json.dumps(await queue.get())
            await websocket.send("hello")
    return fn

async def main():
    queue = asyncio.Queue()
    async with serve(dummy(queue), "localhost", 8765), serve(retran(queue), "localhost", 9000):
        await asyncio.Future()  # run forever

asyncio.run(main())