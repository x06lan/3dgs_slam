import argparse
import asyncio
import json
import logging
import multiprocessing.managers
import multiprocessing.shared_memory
import os
import ssl
import uuid
import cv2
import numpy as np

import multiprocessing

from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay
from av import VideoFrame
import ipdb


class ViewerData:
    def __init__(self) -> None:
        image_shape = (2000, 2000, 3)  # Example shape
        image = np.zeros(image_shape, dtype=np.uint8)
        # shm = multiprocessing.shared_memory.SharedMemory(
        #     create=True, size=image.nbytes)
        self.smm = multiprocessing.managers.SharedMemoryManager()
        self.smm.start()

        self.lock = multiprocessing.Lock()
        self.recive_image_raw = self.smm.SharedMemory(size=image.nbytes)
        self.render_image_raw = self.smm.SharedMemory(size=image.nbytes)

        # dara =[width, height, is_updated]
        self.datas = self.smm.ShareableList(
            [-1, -1, 100, 100, False, False, False, 8, False, 2])
        self.position = self.smm.ShareableList([0.0, 0.0, 0.0])
        self.rotation = self.smm.ShareableList([0.0, 0.0, 0.0])

    @property
    def recive_image(self):
        return np.ndarray((self.datas[1], self.datas[0], 3), dtype=np.uint8, buffer=self.recive_image_raw.buf)

    @property
    def render_image(self):
        return np.ndarray((self.datas[3], self.datas[2], 3), dtype=np.uint8, buffer=self.render_image_raw.buf)

    @property
    def recive_width(self):
        return self.datas[0]

    @recive_width.setter
    def recive_width(self, value):
        self.datas[0] = value

    @property
    def recive_height(self):
        return self.datas[1]

    @recive_height.setter
    def recive_height(self, value):
        self.datas[1] = value

    @property
    def render_width(self):
        return self.datas[2]

    @render_width.setter
    def render_width(self, value):
        self.datas[2] = value

    @property
    def render_height(self):
        return self.datas[3]

    @render_height.setter
    def render_height(self, value):
        self.datas[3] = value

    @property
    def image_update(self):
        return self.datas[4]

    @image_update.setter
    def image_update(self, value):
        self.datas[4] = value

    @property
    def transform_update(self):
        return self.datas[5]

    @transform_update.setter
    def transform_update(self, value):
        self.datas[5] = value

    @property
    def play(self):
        return self.datas[6]

    @play.setter
    def play(self, value):
        self.datas[6] = value

    @property
    def grid(self):
        return self.datas[7]

    @grid.setter
    def grid(self, value):
        self.datas[7] = value

    @property
    def preview(self):
        return self.datas[8]

    @preview.setter
    def preview(self, value):
        self.datas[8] = value

    @property
    def downsample(self):
        return self.datas[9]

    @downsample.setter
    def downsample(self, value):
        self.datas[9] = value

    def require(self):
        self.lock.acquire()

    def release(self):
        self.lock.release()

    def __enter__(self):
        self.require()

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()


ROOT = os.path.dirname(__file__)

logger = logging.getLogger("")
pcs = set()
relay = MediaRelay()


class TrainRenderTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track, data: ViewerData):
        super().__init__()  # don't forget this!
        self.track = track
        self.shareData = data

    async def recv(self):
        frame: VideoFrame = await self.track.recv()
        img = frame.to_ndarray(format="rgb24")

        # with self.shareData:
        self.shareData.require()
        self.shareData.image_update = True

        if (self.shareData.recive_width != img.shape[1] or self.shareData.recive_height != img.shape[0]):
            self.shareData.recive_width = img.shape[1]
            self.shareData.recive_height = img.shape[0]
        share_image = self.shareData.recive_image
        share_image[:] = img[:]

        send_back_image = self.shareData.render_image
        self.shareData.release()

        send_back_image = VideoFrame.from_ndarray(
            send_back_image, format="rgb24")

        send_back_image.pts = frame.pts
        send_back_image.time_base = frame.time_base

        return send_back_image
        # return frame


class RenderTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track, data: ViewerData):
        super().__init__()  # don't forget this!
        self.track = track
        self.shareData = data

    async def recv(self):
        pts, time_base = await self.track.next_timestamp()
        self.shareData.require()
        send_back_image = self.shareData.render_image
        self.shareData.release()

        send_back_image = VideoFrame.from_ndarray(
            send_back_image, format="rgb24")

        send_back_image.pts = pts
        send_back_image.time_base = time_base
        # print("send_back_image")

        return send_back_image


class Viewer:
    def __init__(self, data: ViewerData) -> None:

        self.shareData: ViewerData = data

        self.app = web.Application()
        self.app.on_shutdown.append(self.on_shutdown)
        self.app.router.add_get("/", self.index)
        self.app.router.add_get("/client.js", self.javascript)
        self.app.router.add_post("/offer", self.offer)
        pass

    async def index(self, request):
        content = open(os.path.join(ROOT, "frontend/index.html"), "r").read()
        return web.Response(content_type="text/html", text=content)

    async def javascript(self, request):
        content = open(os.path.join(ROOT, "frontend/client.js"), "r").read()
        return web.Response(content_type="application/javascript", text=content)

    async def on_shutdown(self, app):
        # close peer connections
        coros = [pc.close() for pc in pcs]
        await asyncio.gather(*coros)
        pcs.clear()

    async def offer(self, request):
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        pc = RTCPeerConnection()
        pc_id = "PeerConnection(%s)" % uuid.uuid4()
        pcs.add(pc)

        def log_info(msg, *args):
            logger.info(pc_id + " " + msg, *args)

        log_info("Created for %s", request.remote)

        # prepare local media
        # player = MediaPlayer(os.path.join(ROOT, "demo-instruct.wav"))

        recorder = MediaBlackhole()

        @pc.on("datachannel")
        def on_datachannel(channel):

            @channel.on("message")
            def on_message(message):
                if isinstance(message, str):
                    info = json.loads(message)

                    self.shareData.require()

                    self.transform_update = True

                    self.shareData.position[0] = info["acceleration"][0]
                    self.shareData.position[1] = info["acceleration"][1]
                    self.shareData.position[2] = info["acceleration"][2]

                    self.shareData.rotation[0] = info["rotation"][0]
                    self.shareData.rotation[1] = info["rotation"][1]
                    self.shareData.rotation[2] = info["rotation"][2]

                    self.shareData.grid = info["grid"]
                    self.shareData.play = info["play"]
                    self.shareData.preview = info["preview"]

                    self.shareData.release()

                    # print(self.shareData.position)
                    # data = json.dumps({"position": self.shareData.position})
                    # print(data)
                    # channel.send(data)
                else:
                    raise ValueError("Invalid message")

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            log_info("Connection state is %s", pc.connectionState)
            if pc.connectionState == "failed":
                await pc.close()
                pcs.discard(pc)

        @pc.on("track")
        def on_track(track):
            log_info("Track %s received", track.kind)
            print(track.kind)
            if track.kind == "audio":
                # pc.addTrack(player.audio)
                pc.addTrack(
                    RenderTrack(
                        relay.subscribe(track), data=self.shareData
                    )
                )
                pass
            elif track.kind == "video":
                pc.addTrack(
                    TrainRenderTrack(
                        relay.subscribe(track), data=self.shareData
                    )
                )

            @ track.on("ended")
            async def on_ended():
                log_info("Track %s ended", track.kind)
                await recorder.stop()
            pass
        # handle offer
        await pc.setRemoteDescription(offer)
        await recorder.start()

        # send answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
            ),
        )

    def run(self, host, port,):
        web.run_app(
            self.app, access_log=None, host=host, port=port, ssl_context=None
        )
        pass


if __name__ == "__main__":
    data = ViewerData()
    viwer = Viewer(data)
    viwer.run("0.0.0.0", 8000)
