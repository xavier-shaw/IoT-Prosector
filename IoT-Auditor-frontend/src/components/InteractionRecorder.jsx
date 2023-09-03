import { Button, ButtonGroup, Chip, Dialog, DialogActions, DialogContent, DialogTitle, Grid, TextField, ToggleButton, ToggleButtonGroup } from '@mui/material';
import React, { forwardRef, memo, useCallback, useEffect, useImperativeHandle, useRef, useState } from 'react';
import VideocamIcon from '@mui/icons-material/Videocam';
import VideocamOffIcon from '@mui/icons-material/VideocamOff';
import VisibilityIcon from '@mui/icons-material/Visibility';
import CheckIcon from '@mui/icons-material/Check';
import CancelIcon from '@mui/icons-material/Cancel';
import Webcam from 'react-webcam';
import { v4 as uuidv4 } from "uuid";
import "./InteractionRecorder.css";
import axios from 'axios';

const InteractionRecorder = forwardRef((props, ref) => {
    const { board, chart, createNode, chainNum, setChainNum, status, setStatus } = props;
    const webcamRef = useRef(null);
    const mediaRecorderRef = useRef(null);
    const [camera, setCamera] = useState("");
    const [recording, setRecording] = useState("");
    const [video, setVideo] = useState([]);
    const [newIdx, setNewIdx] = useState(null);
    const [prevIdx, setPrevIdx] = useState(null);
    const [action, setAction] = useState(null);
    const [inputState, setInputState] = useState("");
    const [state, setState] = useState("");
    const [prevState, setPrevState] = useState("");
    const [openVideoDialog, setOpenVideoDiaglog] = useState(false);
    const [openChainDialog, setOpenChainDialog] = useState(false);
    const [openActionDialog, setOpenActionDialog] = useState(false);

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

    useImperativeHandle(ref, () => ({
        setAction,
        setOpenActionDialog
    }))

    useEffect(() => {
        if (recording !== "") {
            setTimeout(() => {
                endRecording();
            }, 5000);   
        }
    }, [recording])

    const handleDataAvailable = (event) => {
        if (event.data.size > 0) {
            setVideo((prev) => prev.concat(event.data));
        }
    };

    const startRecording = async (type) => {
        // start record the video
        setRecording(type);
        setNewIdx(uuidv4());
        const mediaStream = webcamRef.current.stream;
        mediaRecorderRef.current = new MediaRecorder(mediaStream, { mimeType: "video/webm" });
        mediaRecorderRef.current.addEventListener(
            "dataavailable",
            handleDataAvailable
        );
        mediaRecorderRef.current.start();
        // then tell hardware to collect data
        await axios.get(window.HARDWARE_ADDRESS + "/startSensing")
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
                mediaRecorderRef.current.stop();
                if (recording === "state") {
                    createNode(newIdx, status, state, action, prevIdx);
                }
                setOpenVideoDiaglog(true);
            })
    };

    const confirmRecording = () => {
        if (video.length) {
            const blob = new Blob(video, { type: 'video/webm' });
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
                axios
                    .post(window.BACKEND_ADDRESS + "/video/upload", newVideo)
                    .then((resp) => {
                        console.log("upload video success: ", resp);
                    })
                    .catch(error => {
                        console.log("failed with error message: ", error);
                    });
            }

            axios
                .get(window.HARDWARE_ADDRESS + "/storeData", {
                    params: {
                        idx: newIdx,
                        device: board.title,
                    }
                })

            if (recording === "state") {
                setState("");
                setStatus("choose action");
            }
            else {
                setStatus("state");
            }

            setVideo([]);
            setPrevState(state);
            setPrevIdx(newIdx);
            setRecording("");
            setOpenVideoDiaglog(false);
        }
    };

    const cancelRecording = () => {
        setVideo([]);
        setRecording("");
        setOpenVideoDiaglog(false);
    };

    const handleAddBase = () => {
        setChainNum((prev) => (prev + 1));
        setState(inputState);
        setStatus("base state");
        setInputState("");
        setOpenChainDialog(false);
    };

    const handleAddAction = () => {
        setState(inputState);
        setStatus("action");
        setInputState("");
        setOpenActionDialog(false);
    };

    const handleCloseChainDialog = () => {
        setInputState("");
        setOpenChainDialog(false);
    };

    const handleCloseActionDialog = () => {
        setInputState("");
        setOpenActionDialog(false);
    };

    const handleStateChange = (e) => {
        setInputState(e.target.value);
    };

    return (
        <Grid container className='interaction-recorder-div' >
            {/* <h3>Interaction Steps</h3>
            <div className='step-div'>
                {steps.map((step, index) => (
                    <p key={index}>{step}</p>
                ))}
            </div> */}
            <Grid item xs={6} className='full-div'>
                <div className='operation-div'>
                    <div>
                        {status === "start" &&
                            <h4>Please start a new chain</h4>
                        }
                        <Button className="m-2" variant="outlined" color='success' onClick={() => setOpenChainDialog(true)}>Start a new action chain</Button>
                        {status !== "start" &&
                            <>
                                <h6>Action Chain: #{chainNum}</h6>
                                {(() => {
                                    switch (status) {
                                        case "base state": // record a state
                                            return (
                                                <>
                                                    <h6>Current State: {state}</h6>
                                                    <h6>Please start recording state</h6>
                                                </>
                                            );
                                        case "state": // record a state
                                            return (
                                                <>
                                                    <h6>Current State: {prevState}</h6>
                                                    <h6>Please start recording state</h6>
                                                </>
                                            );
                                        case "choose action": // choose an action
                                            return (
                                                <>
                                                    <h6>Current State: {prevState}</h6>
                                                    <h6>Please choose an action</h6>
                                                </>
                                            );
                                        case "action": // record an action
                                            return (
                                                <>
                                                    <h6>Current State: {prevState}</h6>
                                                    <h6>Action: {action}</h6>
                                                    <h6>Next State: {state}</h6>
                                                </>
                                            );
                                        default:
                                            return (
                                                <div></div>
                                            );
                                    }
                                })()}
                            </>
                        }
                    </div>


                    <div>
                        {recording === "state" ?
                            <Button className="mt-2" variant="contained" color="error" onClick={endRecording} startIcon={<VideocamOffIcon />}>End State Recording</Button>
                            :
                            <Button className="mt-2" variant="outlined" disabled={(status !== "state" && status !== "base state") || chainNum === 0} onClick={() => startRecording("state")} startIcon={<VideocamIcon />}>Start State Recording</Button>
                        }
                        {recording === "action" ?
                            <Button className='mt-2' variant="contained" color="error" onClick={endRecording} startIcon={<VideocamOffIcon />}>End Action Recording</Button>
                            :
                            <Button className='mt-2' variant="outlined" disabled={status !== "action"} onClick={() => startRecording("action")} startIcon={<VideocamIcon />}>Start Action Recording</Button>
                        }
                    </div>
                </div>
            </Grid>
            <Grid item xs={6} className='full-div'>
                <div className='full-div'>
                    <Webcam
                        // imageSmoothing={true}
                        audio={recording !== ""}
                        ref={webcamRef}
                        screenshotFormat="image/jpeg"
                        videoConstraints={videoConstraints}
                    />
                </div>
            </Grid>

            <Dialog open={openChainDialog}>
                {/* let user label the state and show user the action
                    highlight the table row when its recording ?
                    show the labeled state and action belowe the camera
                    a recording <=> stop recording "action"/"state" button
                    a start a new chain button => record base state 
                */}
                <DialogTitle>New Action Chain</DialogTitle>
                <DialogContent>
                    <div style={{ padding: "10px" }}>
                        <TextField label="Base State" value={inputState} onChange={handleStateChange} />
                    </div>
                </DialogContent>
                <DialogActions>
                    <Button variant="outlined" color="error" onClick={handleCloseChainDialog}>Cancel</Button>
                    <Button variant="contained" color="primary" onClick={handleAddBase}>Submit</Button>
                </DialogActions>
            </Dialog>

            <Dialog open={openActionDialog}>
                {/* let user label the state and show user the action
                    highlight the table row when its recording ?
                    show the labeled state and action belowe the camera
                    a recording <=> stop recording "action"/"state" button
                    a start a new chain button => record base state 
                */}
                <DialogTitle>Label the Next State</DialogTitle>
                <DialogContent>
                    <div style={{ padding: "10px", display: 'flex', justifyContent: "space-between" }}>
                        <h5 className='me-4'>{prevState}</h5>
                        <Chip className="me-4" label={action} />
                        <TextField size="small" style={{ width: "150px" }} label="Next State" value={inputState} onChange={handleStateChange} />
                    </div>
                </DialogContent>
                <DialogActions>
                    <Button variant="outlined" color="error" onClick={handleCloseActionDialog}>Cancel</Button>
                    <Button variant="contained" color="primary" onClick={handleAddAction}>Submit</Button>
                </DialogActions>
            </Dialog>

            <Dialog open={openVideoDialog && video.length > 0}>
                <DialogTitle>You've complete the {recording} recording.</DialogTitle>
                {/* <DialogContent>Click "confirm" to confirm this interaction.</DialogContent> */}
                <DialogActions>
                    <Button className='mt-2' variant="outlined" color="error" onClick={cancelRecording} startIcon={<CancelIcon />}>Cancel</Button>
                    <Button className='mt-2' variant="contained" color="success" onClick={confirmRecording} startIcon={<CheckIcon />}>Confirm</Button>
                </DialogActions>
            </Dialog>
        </Grid >
    );
});

export default InteractionRecorder;