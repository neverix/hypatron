import numpy as np  # Module that simplifies computations on matrices
from aiohttp import web
import matplotlib.pyplot as plt  # Module used for plotting
from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data
import utils  # Our own utility functions
import time
import subprocess
from muselsl.cli import CLI

import asyncio
from websockets.sync.client import connect
import aiohttp
import requests

from io import BytesIO
from PIL import Image
from base64 import b64encode




# Handy little enum to make code more readable


class Band:
    Delta = 0
    Theta = 1
    Alpha = 2
    Beta = 3


""" EXPERIMENTAL PARAMETERS """
# Modify these to change aspects of the signal processing

# Length of the EEG data buffer (in seconds)
# This buffer will hold last n seconds of data and be used for calculations
BUFFER_LENGTH = 5

# Length of the epochs used to compute the FFT (in seconds)
EPOCH_LENGTH = 1

# Amount of overlap between two consecutive epochs (in seconds)
OVERLAP_LENGTH = 0.8

# Amount to 'shift' the start of each next consecutive epoch
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH

# Index of the channel(s) (electrodes) to be used
# 0 = left ear, 1 = left forehead, 2 = right forehead, 3 = right ear
INDEX_CHANNEL = [0]

async def main():
    """ 0. CONNECT TO SERVER """
    # sock = connect("ws://localhost:8765")

    """ 1. CONNECT TO EEG STREAM """

    # Search for active LSL streams
    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')

    # Set active EEG stream to inlet and apply time correction
    print("Start acquiring data")
    inlet = StreamInlet(streams[0], max_chunklen=12)
    eeg_time_correction = inlet.time_correction()

    # Get the stream info and description
    info = inlet.info()
    description = info.desc()

    # Get the sampling frequency
    # This is an important value that represents how many EEG data points are
    # collected in a second. This influences our frequency band calculation.
    # for the Muse 2016, this should always be 256
    fs = int(info.nominal_srate())

    """ 2. INITIALIZE BUFFERS """

    # Initialize raw EEG data buffer
    eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 1))
    filter_state = None  # for use with the notch filter

    # Compute the number of epochs in "buffer_length"
    n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) /
                              SHIFT_LENGTH + 1))

    # Initialize the band power buffer (for plotting)
    # bands will be ordered: [delta, theta, alpha, beta]
    band_buffer = np.zeros((n_win_test, 4))

    counter = 0
    avgs = []
    metrics = []
    imgs = []
    # avg_samples = 1000
    avg_samples = 80  # for some reason it's really slow, is this data legit?
    batch = 4
    auds = []

    session = aiohttp.ClientSession()
    async def img(req):
        print("req", req)
        print(len(imgs), len(avgs))
        try:
            img = imgs[len(avgs)]
        except IndexError:
            im = Image.new("RGB", (512, 512), (0, 0, 0))
            bio = BytesIO()
            im.resize((128, 128)).save(bio, format="JPEG")
            bio.seek(0)
            img = b64encode(bio.read()).decode("utf-8")
        return web.json_response({"image": img})  # json(val)
    app = web.Application()
    app.add_routes([web.get('/api', img)])
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 3001)
    await site.start()

    """ 3. GET DATA """

    # The try/except structure allows to quit the while loop by aborting the
    # script with <Ctrl-C>
    print('Press Ctrl-C in the console to break the while loop.')
    try:
        # The following loop acquires data, computes band powers, and calculates neurofeedback metrics based on those band powers
        while True:
            # print("Images available", len(imgs), "Metrics available", len(metrics),
                #   "Avgs available", len(avgs))
            if not imgs:
                # roman your stuff goes here
                image_index=None
                if avgs:
                    # print(avgs)
                    # pass
                    # requests.post("...", json=avgs)
                    # sock.send(json.dumps({"averages": avgs}))
                    image_index = np.argmax(np.asarray(avgs)[:, 0])
                    avgs = []
                # imgs = json.loads(sock.recv())
                postf = "" if image_index is None else f"&image_index={image_index}"
                async with session.get(f"https://shreyj1729--eeg-art-root.modal.run/?prompt=cat&negative_prompt={postf}") as pics:
                    imgs = (await pics.json())["images"]
                try:
                    prompts = []
                    for img in imgs:
                        async with session.get(f"https://shreyj1729--eeg-art-image-to-text-dev.modal.run/?image_base64={img}") as texts:
                            prompts.append(await texts.text())
                    print(prompts)
                    auds = []
                    for prompt in prompts:
                        async with session.get(f"https://shreyj1729--eeg-art-audio-webhook-dev.modal.run?prompt={prompt}") as aud:
                            auds.append(aud.read())
                except:
                    pass
                print("sent")
                print("Received:", len(imgs), str(imgs)[:50])
                # batch = len(imgs["images"])
                # imgs = requests.get("...")
            """ 3.1 ACQUIRE DATA """
            # Obtain EEG data from the LSL stream
            eeg_data, timestamp = inlet.pull_chunk(
                timeout=1, max_samples=int(SHIFT_LENGTH * fs))

            # Only keep the channel we're interested in
            try:
                ch_data = np.array(eeg_data)[:, INDEX_CHANNEL]
            except IndexError:
                continue

            # Update EEG buffer with the new data
            eeg_buffer, filter_state = utils.update_buffer(
                eeg_buffer, ch_data, notch=True,
                filter_state=filter_state)

            """ 3.2 COMPUTE BAND POWERS """
            # Get newest samples from the buffer
            data_epoch = utils.get_last_data(eeg_buffer,
                                             EPOCH_LENGTH * fs)

            # Compute band powers
            band_powers = utils.compute_band_powers(data_epoch, fs)
            band_buffer, _ = utils.update_buffer(band_buffer,
                                                 np.asarray([band_powers]))
            # Compute the average band powers for all epochs in buffer
            # This helps to smooth out noise
            smooth_band_powers = np.mean(band_buffer, axis=0)

            # print('Delta: ', band_powers[Band.Delta], ' Theta: ', band_powers[Band.Theta],
            #       ' Alpha: ', band_powers[Band.Alpha], ' Beta: ', band_powers[Band.Beta])

            """ 3.3 COMPUTE NEUROFEEDBACK METRICS """
            # These metrics could also be used to drive brain-computer interfaces

            # Alpha Protocol:
            # Simple redout of alpha power, divided by delta waves in order to rule out noise
            alpha_metric = smooth_band_powers[Band.Alpha] / \
                (smooth_band_powers[Band.Delta] + smooth_band_powers[Band.Alpha])
            # print('Alpha Relaxation: ', alpha_metric)

            # Beta Protocol:
            # Beta waves have been used as a measure of mental activity and concentration
            # This beta over theta ratio is commonly used as neurofeedback for ADHD
            beta_metric = smooth_band_powers[Band.Beta] / \
                smooth_band_powers[Band.Theta]
            # print('Beta Concentration: ', beta_metric)

            # Alpha/Theta Protocol:
            # This is another popular neurofeedback metric for stress reduction
            # Higher theta over alpha is supposedly associated with reduced anxiety
            theta_metric = smooth_band_powers[Band.Theta] / \
                smooth_band_powers[Band.Alpha]
            # print('Theta Relaxation: ', theta_metric)

            plt.clf()
            plt.plot(metrics)
            plt.pause(1e-10)

            metric = [alpha_metric, beta_metric, theta_metric]
            metrics.append(metric)
            if len(metrics) >= avg_samples:  # % avg_samples == avg_samples - 1:
                try:
                    from playsound import playsound
                    open("out.wav", "wb").write(auds[len(avgs)])
                    playsound("out.wav")
                except:
                    pass
                avgs.append(list(np.mean(metrics, axis=0)))
                metrics = []
            if len(avgs) == batch:
                imgs = []

            counter += 1
            await asyncio.sleep(0)

    except KeyboardInterrupt:
        print('Closing!')


if __name__ == "__main__":

    subprocess.Popen(["python3.11", "-m", "muselsl", "stream"])
    time.sleep(12)
    print("Continuing")
    asyncio.run(main())