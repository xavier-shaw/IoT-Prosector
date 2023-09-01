import { Button, ButtonGroup, Chip, Dialog, DialogActions, DialogContent, DialogTitle, TextField, ToggleButton, ToggleButtonGroup } from '@mui/material';
import React, { memo, useCallback, useEffect, useRef, useState } from 'react';
import VideocamIcon from '@mui/icons-material/Videocam';
import VideocamOffIcon from '@mui/icons-material/VideocamOff';
import VisibilityIcon from '@mui/icons-material/Visibility';
import CheckIcon from '@mui/icons-material/Check';
import Webcam from 'react-webcam';
import { v4 as uuidv4 } from "uuid";
import "./InteractionRecorder.css";
import axios from 'axios';

function InteractionRecorder(props) {
    const { board, createNode } = props;
    const webcamRef = useRef(null);
    const mediaRecorderRef = useRef(null);
    const [camera, setCamera] = useState("");
    const [screenshot, setScreenshot] = useState(null);
    const [recording, setRecording] = useState("");
    const [video, setVideo] = useState([]);
    const [newIdx, setNewIdx] = useState(null);
    const [prevIdx, setPrevIdx] = useState(null);
    const [status, setStatus] = useState("Base");
    const [action, setAction] = useState("");
    const [openDialog, setOpenDiaglog] = useState(false);
    const steps = [
        "1. Record the base state whenever you want to start over.",
        "2. Input the action you are going to take.",
        "3. Click \"Start\" to record the action.",
        "4. Remain the action for about 5 seconds until recoding is over.",
        "5. Click \"Confirm\" to confirm the recording and view the action."
    ];
    const cameraWidth = 325;
    const cameraHeight = 325;
    const videoConstraints = {
        deviceId: camera,
        width: cameraWidth,
        height: cameraHeight
    };

    useEffect(() => {
        navigator.mediaDevices.enumerateDevices()
            .then(devices => {
                const videoDevices = devices.filter(device => device.kind === 'videoinput');
                if (videoDevices.length > 0) {
                    let deviceId = "4D39E3577986FFC049FD5845F0A019AEFE2361E2";
                    setCamera(deviceId);
                    console.log("Camera devices: ", videoDevices);
                }
            });
    }, []);

    const capture = () => {
        const imageSrc = webcamRef.current.getScreenshot(); // base64 encoded image. 
        console.log(imageSrc);
        setScreenshot(imageSrc);
    };

    const handleDataAvailable = (event) => {
        if (event.data.size > 0) {
            setVideo((prev) => prev.concat(event.data));
        }
    };

    const startRecording = (type) => {
        // start record the video
        setRecording(type);
        setNewIdx(uuidv4());
        const mediaStream = webcamRef.current.stream;
        mediaRecorderRef.current = new MediaRecorder(mediaStream, { mimeType: "video/mp4" });
        mediaRecorderRef.current.addEventListener(
            "dataavailable",
            handleDataAvailable
        );
        mediaRecorderRef.current.start();
        // then tell hardware to collect data
        axios
            .get(window.HARDWARE_ADDRESS + "/startSensing")
            .then((resp) => {
                console.log(resp.message);
            })
    };

    const endRecording = () => {
        axios
            .get(window.HARDWARE_ADDRESS + "/stopSensing", {
                params: {
                    idx: newIdx,
                    device: board.title,
                }
            })
            .then((resp) => {
                console.log(resp.message);
                mediaRecorderRef.current.stop();
                if (recording === "state") {
                    createNode(newIdx, status, action, status === "Base"? null: prevIdx);
                }
                setOpenDiaglog(true);
                setPrevIdx(newIdx);
            })
    };

    const confirmRecording = () => {
        if (video.length) {
            const blob = new Blob(video, { type: 'video/mp4' });
            // store the data
            const reader = new FileReader();
            reader.readAsDataURL(blob);
            reader.onloadend = function () {
                const videoData = reader.result;
                // Now, you can send `base64data` to your server
                let newVideo = {
                    device: board.title,
                    idx: newIdx,
                    video: videoData
                }
                axios.
                    post(window.BACKEND_ADDRESS + "/video/upload", newVideo)
                    .then((resp) => {
                        console.log("upload video success: ", resp);
                    })
                    .catch(error => {
                        console.log("failed with error message: ", error);
                    });
            }
            // URL.revokeObjectURL(blob);
            setVideo([]);
            setRecording("");
        }
    };

    const handleStatusChange = (e, newStatus) => {
        if (newStatus === "Base") {
            setAction("");
        }
        setStatus(newStatus);
    };

    const handleActionChange = (e) => {
        setAction(e.target.value);
    };

    return (
        <div className='interaction-recorder-div'>
            <h3>Interaction Steps</h3>
            <div className='step-div'>
                {steps.map((step, index) => (
                    <p key={index}>{step}</p>
                ))}
            </div>

            <Webcam
                // imageSmoothing={true}
                audio={recording !== ""}
                ref={webcamRef}
                screenshotFormat="image/jpeg"
                videoConstraints={videoConstraints}
            />
            <br />
            <div className='operation-div'>
                <ToggleButtonGroup
                    value={status}
                    exclusive
                    onChange={handleStatusChange}
                >
                    <ToggleButton value={"Base"} color={status === "Base" ? "primary" : "info"}>Base</ToggleButton>
                    <ToggleButton value={"Action"} color={status === "Action" ? "primary" : "info"}>Action</ToggleButton>
                </ToggleButtonGroup>
                <TextField
                    name="Action"
                    size="small"
                    value={action}
                    disabled={status === "Base"}
                    onChange={handleActionChange}
                    placeholder={status}
                />
            </div>
            {/* take a scrrenshot picture */}
            {/* <Button onClick={capture}>Capture photo</Button> */}
            {/* {screenshot && <img src={screenshot} width={cameraWidth} height={cameraHeight} />} */}

            {/* take a video recording */}
            {recording === "action" ?
                <Button className='mt-2' variant="contained" color="error" onClick={endRecording} startIcon={<VideocamOffIcon />}>End Action Recording</Button>
                :
                <Button className='mt-2' variant="outlined" onClick={() => startRecording("action")} startIcon={<VideocamIcon />}>Start Action Recording</Button>
            }
            {recording === "state" ?
                <Button className="mt-2" variant="contained" color="error" onClick={endRecording} startIcon={<VideocamOffIcon />}>End State Recording</Button>
                :
                <Button className="mt-2" variant="outlined" onClick={() => startRecording("state")} startIcon={<VideocamIcon />}>Start State Recording</Button>
            }

            <Dialog open={openDialog && video.length > 0} onClose={() => { setOpenDiaglog(false) }}>
                <DialogTitle>You've complete the {recording} recording.</DialogTitle>
                {/* <DialogContent>Click "confirm" to confirm this interaction.</DialogContent> */}
                <DialogActions>
                    <Button className='mt-2' variant="outlined" color="success" onClick={confirmRecording} startIcon={<CheckIcon />}>Confirm</Button>
                </DialogActions>
            </Dialog>
        </div>
    );
};

export default memo(InteractionRecorder)