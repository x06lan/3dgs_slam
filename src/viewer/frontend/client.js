// peer connection
var pc = null;

// data channel
var dc = null, dcInterval = null;

let currentTranslation = [0, 0, 0]
let currentRotation= [0, 0, 0]

setInterval(function(){ currentTranslation[0]+=parseInt(joy.GetX())*guiParams.speed*0.0001; }, 50);
setInterval(function(){ currentTranslation[2]+=parseInt(joy.GetY())*guiParams.speed*0.0001;}, 50);

window.addEventListener('deviceorientation', deviceOrientationHandler, false);
function deviceOrientationHandler (eventData) {
    var dir = eventData.alpha+180;
    var tiltFB = eventData.beta+270;
    var tiltLR = eventData.gamma+180;

    currentRotation = [dir, tiltFB, tiltLR]
}
function createPeerConnection(useSTUN) {
    var connection_config = {
        sdpSemantics: 'unified-plan'
    };

    if (useSTUN) {
        connection_config.iceServers = [{ urls: ['stun:stun.l.google.com:19302'] }];
    }

    pc = new RTCPeerConnection(connection_config);


    // connect audio / video
    pc.addEventListener('track', (evt) => {
        if (evt.track.kind == 'video')
            document.getElementById('video').srcObject = evt.streams[0];
        else
            document.getElementById('video').srcObject = evt.streams[0];
            // document.getElementById('audio').srcObject = evt.streams[0];
    });

    return pc;
}

function negotiate(config) {
    return pc.createOffer().then((offer) => {
        return pc.setLocalDescription(offer);
    }).then(() => {
        // wait for ICE gathering to complete
        return new Promise((resolve) => {
            if (pc.iceGatheringState === 'complete') {
                resolve();
            } else {
                function checkState() {
                    if (pc.iceGatheringState === 'complete') {
                        pc.removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }
                }
                pc.addEventListener('icegatheringstatechange', checkState);
            }
        });
    }).then(() => {
        var offer = pc.localDescription;
        var codec;

        codec = config.audioCodec;
        if (codec !== 'Default') {
            offer.sdp = sdpFilterCodec('audio', codec, offer.sdp);
        }

        codec = config.videoCodec;
        if (codec !== 'Default') {
            offer.sdp = sdpFilterCodec('video', codec, offer.sdp);
        }
 

        return fetch('/offer', {
            body: JSON.stringify({
                sdp: offer.sdp,
                type: offer.type,
                video_transform: config.videoEffect,
                preview: config.preview,
            }),
            headers: {
                'Content-Type': 'application/json'
            },
            method: 'POST'
        });
    }).then((response) => {
        return response.json();
    }).then((answer) => {
        return pc.setRemoteDescription(answer);
    }).catch((e) => {
        console.error(e);
        alert(e);
    });
}


function start(config) {
    // config.play = true;
    guiParams.play = true;

    pc = createPeerConnection(config.useStun);

    var time_start = null;

    const current_stamp = () => {
        if (time_start === null) {
            time_start = new Date().getTime();
            return 0;
        } else {
            return new Date().getTime() - time_start;
        }
    };

    if (config.useDataChannel) {
        var parameters = JSON.parse(config.dataChannelOptions);

        dc = pc.createDataChannel('chat', parameters);
        dc.addEventListener('close', () => {
            clearInterval(dcInterval);
            // dataChannelLog.textContent += '- close\n';
        });
        dc.addEventListener('open', () => {
            // dataChannelLog.textContent += '- open\n';
            // let toInt= (x) => { x? 1: 0}
            dcInterval = setInterval(() => {

                let message = {
                    "time":current_stamp(),
                    "acceleration": currentTranslation,
                    "rotation": currentRotation,
                    "downsample": guiParams.downsample,
                    "grid": guiParams.grid,
                    "preview": guiParams.preview,
                    "play": guiParams.play,
                };
                // console.log(message)

                dc.send(JSON.stringify(message));
            }, 10);
        });
        dc.addEventListener('message', (evt) => {
            // dataChannelLog.textContent += '< ' + evt.data + '\n';
            // let data = evt.data.split(',')
            let data = JSON.parse(evt.data)
            console.log(data)
            // for (let i = 0; i < data.length && i < 3; i++) {
            //     currentTranslation[i] = parseFloat(data[i])
            // }

        });
    }

    // Build media constraints.

    const constraints = {
        audio: config.useAudio,
        video: config.useVideo
    };

    if (config.useAudio) {
        const audioConstraints = {};

        const device = config.audioDevice;
        if (device) {
            audioConstraints.deviceId = { exact: device };
        }

        constraints.audio = Object.keys(audioConstraints).length ? audioConstraints : true;
    }

    if (config.useVideo && !config.preview) {
        const videoConstraints = {
            width: { min: 320, ideal: 1280, max: 2560 },
            height: { min: 240, ideal: 720, max: 1440 },
            facingMode: "environment"
        };

        const device = config.videoDevice;
        const resolution = config.videoResolution;
        constraints.video = Object.keys(videoConstraints).length ? videoConstraints : true;
    }

    // Acquire media and start negociation.

    if (constraints.audio || constraints.video ) {
        navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
            stream.getTracks().forEach((track) => {
                pc.addTrack(track, stream);
            });
            return negotiate(config);
        }, (err) => {
            console.log(err)
            alert('Could not acquire media: ' + err);
        });
    } else {
        negotiate(config);
    }
}

function stop(config) {
    // config.play = false;
    guiParams.play = false;
    

    // close data channel
    if (dc) {
        dc.close();
    }

    // close transceivers
    if (pc.getTransceivers) {
        pc.getTransceivers().forEach((transceiver) => {
            if (transceiver.stop) {
                transceiver.stop();
            }
        });
    }

    // close local audio / video
    pc.getSenders().forEach((sender) => {
        sender.track.stop();
    });

    // close peer connection
    setTimeout(() => {
        pc.close();
    }, 500);
}

function sdpFilterCodec(kind, codec, realSdp) {
    var allowed = []
    var rtxRegex = new RegExp('a=fmtp:(\\d+) apt=(\\d+)\r$');
    var codecRegex = new RegExp('a=rtpmap:([0-9]+) ' + escapeRegExp(codec))
    var videoRegex = new RegExp('(m=' + kind + ' .*?)( ([0-9]+))*\\s*$')

    var lines = realSdp.split('\n');

    var isKind = false;
    for (var i = 0; i < lines.length; i++) {
        if (lines[i].startsWith('m=' + kind + ' ')) {
            isKind = true;
        } else if (lines[i].startsWith('m=')) {
            isKind = false;
        }

        if (isKind) {
            var match = lines[i].match(codecRegex);
            if (match) {
                allowed.push(parseInt(match[1]));
            }

            match = lines[i].match(rtxRegex);
            if (match && allowed.includes(parseInt(match[2]))) {
                allowed.push(parseInt(match[1]));
            }
        }
    }

    var skipRegex = 'a=(fmtp|rtcp-fb|rtpmap):([0-9]+)';
    var sdp = '';

    isKind = false;
    for (var i = 0; i < lines.length; i++) {
        if (lines[i].startsWith('m=' + kind + ' ')) {
            isKind = true;
        } else if (lines[i].startsWith('m=')) {
            isKind = false;
        }

        if (isKind) {
            var skipMatch = lines[i].match(skipRegex);
            if (skipMatch && !allowed.includes(parseInt(skipMatch[2]))) {
                continue;
            } else if (lines[i].match(videoRegex)) {
                sdp += lines[i].replace(videoRegex, '$1 ' + allowed.join(' ')) + '\n';
            } else {
                sdp += lines[i] + '\n';
            }
        } else {
            sdp += lines[i] + '\n';
        }
    }

    return sdp;
}

function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); // $& means the whole matched string
}

function threejs() {
    const myCanvas = document.querySelector('#three');

    const axes = new THREE.AxesHelper();
    const phone = new THREE.AxesHelper();

    const scene = new THREE.Scene();
    const grid = new THREE.GridHelper(100, 100);

    const points = [];
    points.push(new THREE.Vector3(0, 0, 0));
    points.push(new THREE.Vector3(0, 0, 0));

    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({ color: 0x0000ff });
    const line = new THREE.Line(geometry, material);

    scene.add(grid);
    // scene.add(axes);
    scene.add(phone);
    scene.add(line);
    // scene.add(cameraOrthoHelper);

    const viewCamera = new THREE.PerspectiveCamera(
        50,
        4.0 / 3.0,
        0.1,
        1000,
    );

    viewCamera.position.set(10, 10, 10);
    viewCamera.lookAt(scene.position);

    const renderer = new THREE.WebGLRenderer({
        antialias: true,
        canvas: myCanvas,
        outerWidth: window.innerWidth * 0.5,
    });
    renderer.setClearColor(0x000000, 1.0); // 背景色
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(myCanvas.offsetWidth, myCanvas.offsetHeight);

    const orbitControls = new THREE.OrbitControls(
        viewCamera,
        renderer.domElement
    );
    renderer.setAnimationLoop(() => {
        // cameraOrthoHelper.position.x = currentTranslation[0]
        // currentRotation[1] += 0.01
        // cameraOrthoHelper.rotation.y = currentRotation[1]
        // axes.rotation.y = currentRotation[1]
        // cameraOrthoHelper.update();

        points.push(new THREE.Vector3(currentTranslation[0], currentTranslation[1], currentTranslation[2]));
        // console.log(currentTranslation)
        geometry.setFromPoints(points);
        // lineGeometry.verticesNeedUpdate = true;

        phone.position.set(currentTranslation[0], currentTranslation[1], currentTranslation[2]);
        // phone.rotation.set(currentRotation[0], currentRotation[1], currentRotation[2]);

        // console.log(cameraOrthoHelper.rotation)

        orbitControls.update();
        renderer.render(scene, viewCamera);
    });
    return scene

}


// enumerateInputDevices();
// threejs();
