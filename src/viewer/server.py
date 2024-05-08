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


class ViewerData:
    def __init__(self) -> None:
        image_shape = (2000, 2000, 3)  # Example shape
        image = np.zeros(image_shape, dtype=np.uint8)
        # shm = multiprocessing.shared_memory.SharedMemory(
        #     create=True, size=image.nbytes)
        self.smm = multiprocessing.managers.SharedMemoryManager()
        self.smm.start()

        self.lock = multiprocessing.Lock()
        self.image_raw = self.smm.SharedMemory(size=image.nbytes)

        # dara =[width, height, is_updated]
        self.datas = self.smm.ShareableList([-1, -1, False])
        self.position = self.smm.ShareableList([0.0, 0.0, 0.0])
        self.rotation = self.smm.ShareableList([0.0, 0.0, 0.0])

    @property
    def image(self):
        return np.ndarray((self.datas[1], self.datas[0], 3), dtype=np.uint8, buffer=self.image_raw.buf)

    @property
    def width(self):
        return self.datas[0]

    @width.setter
    def width(self, value):
        self.datas[0] = value

    @property
    def height(self):
        return self.datas[1]

    @height.setter
    def height(self, value):
        self.datas[1] = value

    @property
    def image_update(self):
        return self.datas[2]

    @image_update.setter
    def image_update(self, value):
        self.datas[2] = value

    def require(self):
        self.lock.acquire()

    def release(self):
        self.lock.release()


ROOT = os.path.dirname(__file__)

logger = logging.getLogger("")
pcs = set()
relay = MediaRelay()


class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track, transform, data: ViewerData):
        super().__init__()  # don't forget this!
        self.track = track
        self.transform = transform
        self.shareData = data

    async def recv(self):
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")

        self.shareData.require()

        if (self.shareData.width != img.shape[1] or self.shareData.height != img.shape[0]):
            # setting width and height
            self.shareData.width = img.shape[1]
            self.shareData.height = img.shape[0]

        self.shareData.image_update = True
        share_image = self.shareData.image
        share_image[:] = img[:]

        self.shareData.release()
        return frame


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
                    position = f"{self.shareData.position[0]},{self.shareData.position[1]},{self.shareData.position[2]}"
                    # print(f"position: {position}")
                    channel.send(position)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            log_info("Connection state is %s", pc.connectionState)
            if pc.connectionState == "failed":
                await pc.close()
                pcs.discard(pc)

        @pc.on("track")
        def on_track(track):
            log_info("Track %s received", track.kind)

            if track.kind == "audio":
                # pc.addTrack(player.audio)
                recorder.addTrack(track)
            elif track.kind == "video":
                pc.addTrack(
                    VideoTransformTrack(
                        relay.subscribe(track), transform=params["video_transform"], data=self.shareData
                    )
                )
                # if args.record_to:
                #     recorder.addTrack(relay.subscribe(track))

            @ track.on("ended")
            async def on_ended():
                log_info("Track %s ended", track.kind)
                await recorder.stop()

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

    pass
