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
    const { board, chart, createNode, chainNum, setChainNum, status, setStatus, curNode, prevNode } = props;
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
                let deviceId;
                for (const [idx, device] of Object.entries(videoDevices)) {
                    if (device.label === "1080P Pro Stream (046d:0894)") {
                        deviceId = device.id;
                    }
                }
                deviceId = videoDevices[0].deviceId;
                if (videoDevices.length > 0) {
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
        if (recording === "state") {
            setTimeout(() => {
                endRecording();
            }, 5300);
        }
        else if (recording === "action") {
            setTimeout(() => {
                endRecording();
            }, 2500);
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
        let newIdx = uuidv4();
        setNewIdx(newIdx);
        const mediaStream = webcamRef.current.stream;
        mediaRecorderRef.current = new MediaRecorder(mediaStream, { mimeType: "video/webm" });
        mediaRecorderRef.current.addEventListener(
            "dataavailable",
            handleDataAvailable
        );
        mediaRecorderRef.current.start();
        // then tell hardware to collect data
        if (type === "state") {
            await axios.get(window.HARDWARE_ADDRESS + "/startSensing", {
                params: {
                    device: board.title,
                    idx: newIdx
                }
            })
        }
    };

    const endRecording = () => {
        axios
            .get(window.HARDWARE_ADDRESS + "/stopSensing")
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
            
            const reader = new FileReader();
            reader.readAsDataURL(blob);
            reader.onloadend = function () {
                const videoData = reader.result;
                
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

            if (recording === "state") {
                setState("");
                setStatus("choose action");
                axios
                    .get(window.HARDWARE_ADDRESS + "/storeData", {
                        params: {
                            device: board.title,
                            idx: newIdx
                        }
                    })
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
        setStatus("base state");
    };

    const handleAddAction = () => {
        setState(inputState);
        setInputState("");
        setOpenActionDialog(false);
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
            <Grid item xs={6} className='full-div'>
                <div className='operation-div'>
                    <div>
                        {/* {status === "start" &&
                            <h4 style={{ fontFamily: "Times New Roman" }}>Please label the state you begin with.</h4>
                        } */}
                        <Button variant="outlined" color='success' disabled={status !== "start"} sx={{ fontWeight: "bold", fontSize: 18, fontFamily: "Times New Roman" }} onClick={() => handleAddBase()}>
                            begin with a new state
                        </Button>
                        {status !== "start" &&
                            <>
                                <h4 style={{ fontFamily: "Times New Roman" }}>Chain: #{chainNum}</h4>
                                {(() => {
                                    console.log("here", chart.nodes);
                                    switch (status) {
                                        case "base state": // record a state
                                            return (
                                                <>
                                                    {/* <h4 style={{ fontFamily: "Times New Roman" }}>Current State: {state}</h4> */}
                                                    <h4 style={{ fontFamily: "Times New Roman", fontWeight: "bold" }}>Please start recording the state.</h4>
                                                </>
                                            );
                                        case "state": // record a state
                                            return (
                                                <>
                                                    <h4 style={{ fontFamily: "Times New Roman" }}>Current State: {chart.nodes[chart.nodes.length - 1].data.label}</h4>
                                                    <h4 style={{ fontFamily: "Times New Roman", fontWeight: "bold" }}>Please start recording state.</h4>
                                                </>
                                            );
                                        case "choose action": // choose an action
                                            return (
                                                <>
                                                    <h4 style={{ fontFamily: "Times New Roman" }}>Current State: {chart.nodes[chart.nodes.length - 1].data.label}</h4>
                                                    <h4 style={{ fontFamily: "Times New Roman", fontWeight: "bold" }}>Please choose an action.</h4>
                                                </>
                                            );
                                        case "action": // record an action
                                            return (
                                                <>
                                                    <h4 style={{ fontFamily: "Times New Roman" }}>Current State: {chart.nodes[chart.nodes.length - 1].data.label}</h4>
                                                    <h4 style={{ fontFamily: "Times New Roman", fontWeight: "bold" }}>Action: {action}</h4>
                                                    {/* <h4 style={{ fontFamily: "Times New Roman" }}>Next State: {state}</h4> */}
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
                            <Button variant="contained" sx={{ fontWeight: "bold", fontSize: 20, fontFamily: "Times New Roman" }}
                                startIcon={<VideocamOffIcon />}>
                                Recording
                            </Button>
                            :
                            <Button variant="outlined" disabled={(status !== "state" && status !== "base state") || chainNum === 0}
                                sx={{ fontWeight: "bold", fontSize: 20, fontFamily: "Times New Roman" }} onClick={() => startRecording("state")} startIcon={<VideocamIcon />}>
                                Start State Recording
                            </Button>
                        }
                        {recording === "action" ?
                            <Button className='mt-2' variant="contained" sx={{ fontWeight: "bold", fontSize: 20, fontFamily: "Times New Roman" }}
                                startIcon={<VideocamOffIcon />}>
                                Recording
                            </Button>
                            :
                            <Button className='mt-2' variant="outlined" disabled={status !== "action"} sx={{ fontWeight: "bold", fontSize: 20, fontFamily: "Times New Roman" }}
                                onClick={() => startRecording("action")} startIcon={<VideocamIcon />}>
                                Start Action Recording
                            </Button>
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

            {/* <Dialog open={openChainDialog}>
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
            </Dialog> */}

            <Dialog open={openActionDialog}>
                <DialogTitle>Annotate the State You Just Record</DialogTitle>
                <DialogContent>
                    <div style={{ padding: "10px", display: 'flex', justifyContent: "space-between" }}>
                        {status !== "base state" &&
                            <h5 className='me-4'>{prevState}</h5> &&
                            <Chip className="me-4" label={action} />
                        }
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
                {recording === "state" && <DialogContent>Please annotate the state you just record.</DialogContent>}
                <DialogActions>
                    {/* <Button className='mt-2' variant="outlined" color="error" onClick={cancelRecording} startIcon={<CancelIcon />}>Cancel</Button> */}
                    <Button className='mt-2' variant="contained" color="success" onClick={confirmRecording} startIcon={<CheckIcon />}>Confirm</Button>
                </DialogActions>
            </Dialog>
        </Grid >
    );
});

export default InteractionRecorder;