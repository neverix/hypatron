import asyncio
from websockets.server import serve
from io import BytesIO
from PIL import Image
from base64 import b64encode
import json


def b64enc(im):
    bio = BytesIO()
    im.save(bio, format="PNG")
    return b64encode(bio.read()).decode("utf-8")


async def dummy(websocket):
    while True:
        im = Image.new("RGB", (512, 512))
        to_send = {"images": [b64enc(im)] * 4}
        await websocket.send(json.dumps(to_send))
        print("sent", str(to_send)[:50])
        avgs = json.loads(await websocket.recv())
        print(avgs)

async def main():
    async with serve(dummy, "localhost", 8765):
        await asyncio.Future()  # run forever

asyncio.run(main())