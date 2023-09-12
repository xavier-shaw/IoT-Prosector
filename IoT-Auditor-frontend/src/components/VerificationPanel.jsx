import React, { forwardRef, useImperativeHandle, useState, useEffect } from "react";
import OnlinePredictionIcon from '@mui/icons-material/OnlinePrediction';
import { Button, Dialog, DialogActions, DialogTitle, FormControl, InputLabel, LinearProgress, MenuItem, Select } from "@mui/material";
import "./VerificationPanel.css";
import axios from "axios";
import { v4 as uuidv4 } from "uuid";

const VerificatopmPanel = forwardRef((props, ref) => {
    const { board, chart, status, setStatus, verifyState, predictState } = props;
    const [action, setAction] = useState("None");
    const [predicting, setPredicting] = useState(false);
    const [doingAction, setDoingAction] = useState(false);
    const [readyForNextAction, setReadyForNextAction] = useState(false);
    const [wrongPrediction, setWrongPrediction] = useState(false);
    const [correctState, setCorrectState] = useState("");
    const [stateIdx, setStateIdx] = useState(null);

    useImperativeHandle(ref, () => ({
        setAction,
        setDoingAction,
        endStatePrediction
    }));

    useEffect(() => {
        if (predicting) {
            setTimeout(async () => {
                await verifyState(stateIdx);
            }, 5300);
        }
    }, [predicting])

    const startStatePrediction = () => {
        if (!predicting) {
            setPredicting(true);
            let newIdx =  uuidv4();
            setStateIdx(newIdx);
            axios.get(window.HARDWARE_ADDRESS + "/startSensing", {
                params: {
                    device: board.title,
                    idx: newIdx
                }
            })   
        }
    };

    const endStatePrediction = () => {
        setReadyForNextAction(false);
        setStatus("verifying");
        setPredicting(false);
    };

    const onFinishAction = () => {
        setDoingAction(false);
        setStatus("state");
    };

    const onPredictionCorrect = () => {
        axios
        .get(window.HARDWARE_ADDRESS + "/verify", {
            params: {
                device: board.title,
                predict: predictState.data.label,
                correct: predictState.data.label 
            }
        })
        .then((resp) => {
            setReadyForNextAction(true);
            setStatus("choose action");
        }) 
    };

    const onPredictionWrong = () => {
        setWrongPrediction(true);
    };

    const handleSelectChange = (evt) => {
        setCorrectState(evt.target.value);
    };

    const submitCorrectState = () => {
        axios
        .get(window.HARDWARE_ADDRESS + "/verify", {
            params: {
                predict: predictState.data.label,
                correct: correctState 
            }
        })
        .then((resp) => {
            setWrongPrediction(false);
            setReadyForNextAction(true);
            setCorrectState("");
            setStatus("choose action");
        })
    }

    return (
        <div className="verification-panel-div">
            <h4>Verification</h4>
            <div className="operation-div">
                <div>
                    {(() => {
                        switch (status) {
                            case "start": // record a state
                                return (
                                    <>
                                        <h3 style={{ fontFamily: "Times New Roman", fontWeight: "bold",  }}>Please start a state prediction.</h3>
                                    </>
                                );
                            case "state": // record a state
                                return (
                                    <>
                                        <h4 style={{ fontFamily: "Times New Roman" }}>Your Action is: {action}</h4>
                                        <h4 style={{ fontFamily: "Times New Roman", fontWeight: "bold",  }}>Please start the state prediction.</h4>
                                    </>
                                );
                            case "verifying":
                            case "choose action": // choose an action
                                return (
                                    <>
                                        <h4 style={{ fontFamily: "Times New Roman" }}>Your Action is: {action}</h4>
                                        <h4 style={{ fontFamily: "Times New Roman" }}>Current Predicted State is: {predictState?.data?.label}</h4>
                                        {readyForNextAction &&
                                            <h4 style={{ fontFamily: "Times New Roman", fontWeight: "bold" }}>Please choose the next action.</h4>
                                        }
                                    </>
                                );
                            default:
                                return (
                                    <>
                                    </>
                                );
                        }
                    })()}
                </div>

                <div>
                    <Button variant={predicting ? "contained" : "outlined"} disabled={(status !== "state" && status !== "start")}
                        sx={{ fontWeight: "bold", fontSize: 20, fontFamily: "Times New Roman" }} onClick={startStatePrediction} startIcon={<OnlinePredictionIcon />}>
                        {predicting? "Predicting" : "Start State Prediction"}
                    </Button>
                    <p style={{fontFamily: "Times New Roman", fontSize: 20}}>It takes up to 30 seconds to finish the prediction.</p>
                    <LinearProgress/>
                </div>
            </div>

            {status === "verifying" &&
                <div>
                    {!wrongPrediction &&
                        <div>
                            <h4>Is the prediction of your model correct?</h4>
                            <Button className="me-5" variant="outlined" color="success" onClick={onPredictionCorrect}>Yes</Button>
                            <Button variant="outlined" color="error" onClick={onPredictionWrong}>No</Button>
                        </div>
                    }
                    {wrongPrediction &&
                        <div>
                            <h4>Please select the correct state: </h4>
                            <FormControl>
                                <InputLabel>Correct State</InputLabel>
                                <Select
                                    value={correctState}
                                    onChange={handleSelectChange}
                                    label="Correct State"
                                    sx={{width: "200px"}}
                                >
                                    {chart?.nodes?.filter((n) => !n.parentNode).map((node, i) => (
                                        <MenuItem key={i} value={node.data.representLabel}>{node.data.representLabel}</MenuItem>
                                    ))}
                                </Select>
                            </FormControl>
                            <Button className="ms-5" variant="outlined" color="primary" onClick={submitCorrectState}>Submit</Button>
                        </div>
                    }
                </div>
            }

            <Dialog open={doingAction}>
                <DialogTitle>Please finish the action: {action}</DialogTitle>
                <DialogActions>
                    <Button variant="outlined" color="success" onClick={onFinishAction}>Finished</Button>
                </DialogActions>
            </Dialog>
        </div>
    )
});

export default VerificatopmPanel;