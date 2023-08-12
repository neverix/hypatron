
from gradio import networking
import secrets
host = "0.0.0.0"
port = "3000"
share_token = secrets.token_urlsafe(32)
share_link = networking.setup_tunnel(host, port, share_token)
print(f"Shared at {share_link}")
from time import sleep
while True:
    sleep(10)