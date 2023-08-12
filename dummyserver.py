import asyncio
from aiohttp import web
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
            im = Image.new("RGB", (512, 512), (0, 0, 0))
            to_send = {"images": [b64enc(im)] * 4}
            await queue.put(to_send)
            await websocket.send(json.dumps(to_send))
            print("sent", str(to_send)[:50])
            avgs = json.loads(await websocket.recv())
            print(avgs)
    return fn


async def main():
    queue = asyncio.Queue()
    async with serve(dummy(queue), "localhost", 8765):
        for _ in range(10):
            await queue.put({"images": []})
        await asyncio.Event().wait()

asyncio.run(main())