import { Button, ButtonGroup, Chip, TextField, ToggleButton, ToggleButtonGroup } from '@mui/material';
import React, { memo, useCallback, useEffect, useRef, useState } from 'react';
import VideocamIcon from '@mui/icons-material/Videocam';
import VideocamOffIcon from '@mui/icons-material/VideocamOff';
import VisibilityIcon from '@mui/icons-material/Visibility';
import CheckIcon from '@mui/icons-material/Check';
import Webcam from 'react-webcam';
import "./InteractionRecorder.css";

function InteractionRecorder(props) {
    const webcamRef = useRef(null);
    const mediaRecorderRef = useRef(null);
    const [camera, setCamera] = useState("");
    const [screenshot, setScreenshot] = useState(null);
    const [recording, setRecording] = useState(false);
    const [video, setVideo] = useState([]);
    const [videoUrl, setVideoUrl] = useState(null);
    const [preview, setPreview] = useState(false);
    const [status, setStatus] = useState("Base");
    const [action, setAction] = useState("");
    const steps = [
        "1. Record the base state whenever you want to start over.",
        "2. Input the action you are going to take.",
        "3. Click \"Start\" to record the action.",
        "4. Remain the action for about 5 seconds until recoding is over.",
        "5. Click \"Confirm\" to confirm the recording and view the action."
    ]
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
                    setCamera(videoDevices[0].deviceId);
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

    const startRecording = () => {
        // start record the video
        setRecording(true);
        const mediaStream = webcamRef.current.stream;
        mediaRecorderRef.current = new MediaRecorder(mediaStream, { mimeType: "video/mp4" });
        mediaRecorderRef.current.addEventListener(
            "dataavailable",
            handleDataAvailable
        );
        mediaRecorderRef.current.start();

        // then tell hardware to collect data => wait 1 second to mitigate human latency
        setTimeout(() => {
            axios
                .get(window.HARDWARE_ADDRESS + "/powerSensing")
                .then((resp) => {
                    console.log("power data", resp.data);
                    endRecording();
                })
        }, 1000);
    };

    const endRecording = () => {
        mediaRecorderRef.current.stop();
        setRecording(false);
    };

    const confirmRecording = () => {
        if (video.length) {
            setPreview(true);
            const blob = new Blob(video, { type: 'video/mp4' });
            console.log(blob)
            setVideoUrl(URL.createObjectURL(blob));
            URL.revokeObjectURL(blob);
            setVideo([]);
        }
    };

    const handleStatusChange = (e, newStatus) => {
        setStatus(newStatus);
    };

    const handleActionChange = (e) => {
        setAction(e.target.value);
    };

    return (
        <div className='interaction-recorder-div'>
            <h3>Interaction Stage</h3>
            <div className='step-div'>
                {steps.map((step, index) => (
                    <p key={index}>{step}</p>
                ))}
            </div>

            <Webcam
                // imageSmoothing={true}
                audio={recording}
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
                    placeholder="Action"
                />
            </div>
            {/* take a scrrenshot picture */}
            {/* <Button onClick={capture}>Capture photo</Button> */}
            {/* {screenshot && <img src={screenshot} width={cameraWidth} height={cameraHeight} />} */}

            {/* take a video recording */}
            {recording ?
                <Button variant="contained" color="error" onClick={endRecording} startIcon={<VideocamOffIcon />}>Stop Recording</Button>
                :
                <Button variant="outlined" onClick={startRecording} startIcon={<VideocamIcon />}>Start Recording</Button>
            }
            <br />
            {!recording && video.length > 0 && <Button className='mt-2' variant="outlined" color="success" onClick={confirmRecording} startIcon={<CheckIcon />}>Confirm</Button>}
            {/* {preview && <Button variant="contained" color="success" startIcon={<CheckIcon />}>Confirm</Button>} */}
            {/* <video src={videoUrl} controls width={cameraWidth} height={cameraHeight} /> */}
        </div>
    );
};

export default memo(InteractionRecorder)