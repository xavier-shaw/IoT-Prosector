import React, { useRef } from "react";
import MenuBar from "../components/MenuBar";
import { Helmet } from 'react-helmet';
import "./board.css";
import axios from "axios";
import { useState } from "react";
import { useEffect } from "react";
import { cloneDeep } from 'lodash';
import { useParams } from "react-router-dom";
import NodeChart from "../components/NodeChart";
import TimelineChart from "../components/TimelineChart";
import { MarkerType } from "reactflow";
import { Button, Chip } from "@mui/material";
import SideNodeBar from "../components/SideNodeBar";

export default function Board(props) {
    let params = useParams();
    const [board, setBoard] = useState({});
    const [step, setStep] = useState(0);
    const stages = ["Exploration", "Annotation", "Verification"];
    const hints = [
        "Click button to start or stop generating state diagram.",
        "Click a node or edge to annotate, then confirm the annotation.",
        "Click button to begin your verification."
    ]
    const [states, setStates] = useState([]);
    const [totalStates, setTotalStates] = useState([]);
    const [transitions, setTransitions] = useState([]);
    const [isSensing, setIsSensing] = useState(-1);
    const nodeChartRef = useRef(null);
    const ref = useRef({});

    useEffect(() => {
        axios
            .get(window.BACKEND_ADDRESS + "/boards/" + params.boardId)
            .then((resp) => {
                let board_data = resp.data[0];
                board_data.data = JSON.parse(board_data.data);
                console.log("board data", board_data);
                setBoard(board_data);
                getTotalStates(board_data.title);
                axios
                    .get(window.BACKEND_ADDRESS + "/shared/update/" + board_data.title)
                    .then((resp) => {
                        console.log("device", resp.data);
                    })
            });
        return () => {
        }
    }, [params.boardId]);

    useEffect(() => {
        if (board.hasOwnProperty("data")) {
            if (board.data.hasOwnProperty("statesDict")) {
                let states = [];
                for (const [idx, state] of Object.entries(board.data.statesDict)) {
                    states.push(state);
                };
                setStates(states);
            };

            if (board.data.hasOwnProperty("transitionsDict")) {
                let transitions = [];
                for (const [idx, transition] of Object.entries(board.data.transitionsDict)) {
                    transitions.push(transition);
                };
                setTransitions(transitions);
            };
        }
    }, [board]);

    useEffect(() => {
        let totalStatesCpy = cloneDeep(totalStates);
        for (const s of states) {
            for (const state of totalStatesCpy) {
                if (s.id === state.id) {
                    state.data.label = s.data.label;
                }
            }
        };
        setTotalStates(totalStatesCpy);
    }, [states])

    const getTotalStates = (title) => {
        axios.
            get(window.BACKEND_ADDRESS + "/states/" + title)
            .then((resp) => {
                let iotStates = resp.data;
                let totalStates = [];

                for (const iotState of iotStates) {
                    // record this state's information
                    totalStates.push({
                        id: "node_" + iotState.idx,
                        time: iotState.time,
                        data: { label: "State " + iotState.idx }
                    });
                }

                setTotalStates(totalStates);
            })
    };

    const handleClickNext = () => {
        if (step === 0) {
            axios.
                get(window.BACKEND_ADDRESS + "/datas/" + board.title)
                .then((resp) => {
                    console.log("iot data: ", resp.data);
                });
        };
        setStep((prevStep) => (prevStep + 1));
    };

    const handleClickBack = () => {
        setStep((prevStep) => (prevStep - 1));
    }

    const handleTitleFocusOut = (titleText) => {
        console.log("Update board title:", titleText);
        setBoard(prevBoard => ({ ...prevBoard, title: titleText }));
    };

    const deletePrevHightlight = (prevStateIdx, curStateIdx) => {
        let prevStateNode = document.getElementById("node_" + curStateIdx);
        prevStateNode.style.backgroundColor = "white";
        let prevTransitionLabel = document.getElementById("edge_" + prevStateIdx + "-" + curStateIdx + "_label");
        prevTransitionLabel.style.backgroundColor = "rgba(0, 0, 0, 0.08)";
    };

    const addNewHighlight = (curStateIdx, nextStateIdx) => {
        let nextStateNode = document.getElementById("node_" + nextStateIdx);
        let nextTransitionEdge = document.getElementById("edge_" + curStateIdx + "-" + nextStateIdx + "_label");
        nextStateNode.style.backgroundColor = "skyblue";
        nextTransitionEdge.style.backgroundColor = "skyblue";
    };

    const onSave = () => {
        updateAnnotation();
        let boardCpy = cloneDeep(board);
        let boardData = boardCpy.data;
        let statesDict = boardData.hasOwnProperty("statesDict") ? boardData.statesDict : {};
        let transitionsDict = boardData.hasOwnProperty("transitionsDict") ? boardData.transitionsDict : {};
        for (const state of states) {
            let state_id = state["id"];
            statesDict[state_id] = state;
        };

        for (const transition of transitions) {
            let transition_id = transition["id"];
            transitionsDict[transition_id] = transition;
        };

        boardData.statesDict = statesDict;
        boardData.transitionsDict = transitionsDict;
        setBoard((prevBoard) => ({ ...prevBoard, data: boardData }));
        console.log("update board data", boardCpy);
        boardCpy["data"] = JSON.stringify(boardData);
        axios
            .post(window.BACKEND_ADDRESS + "/boards/saveBoard", { boardId: board._id, updates: boardCpy })
            .then((resp) => {
                console.log("update board", resp.data);
            });
    };

    const updateAnnotation = () => {
        nodeChartRef.current.updateAnnotation();
    };

    const handleClickAnnotate = () => {
        if (isSensing !== -1) {
            axios
                .get(window.BACKEND_ADDRESS + "/shared/stop")
                .then((resp) => {
                    console.log("stop sensing at annotation stage", resp.data);
                    clearInterval(isSensing);
                    setIsSensing(-1);
                });
            updateAnnotation();
        }
        else {
            axios
                .get(window.BACKEND_ADDRESS + "/shared/start/" + "annotation")
                .then((resp) => {
                    console.log("start sensing at annotation stage", resp.data);
                    let idx = setInterval(() => {
                        annotationSensing();
                    }, 1000);
                    setIsSensing(idx);
                })
        }
    };

    const handleClickExplore = () => {
        if (isSensing !== -1) {
            axios
                .get(window.BACKEND_ADDRESS + "/shared/stop")
                .then((resp) => {
                    console.log("stop sensing at exploration stage", resp.data);
                    clearInterval(isSensing);
                    setIsSensing(-1);
                })
        }
        else {
            axios
                .get(window.BACKEND_ADDRESS + "/shared/start/" + "exploration")
                .then((resp) => {
                    console.log("start sensing at exploration stage", resp.data);
                    let idx = setInterval(() => {
                        sensing();
                    }, 1000);
                    setIsSensing(idx);
                })
        }
    };

    const sensing = () => {
        axios.
            get(window.BACKEND_ADDRESS + "/states/" + board.title)
            .then((resp) => {
                console.log("iot states", resp.data);
                let iotStates = resp.data;
                let boardData = board.data;
                let states = board.data.hasOwnProperty("statesDict") ? board.data.statesDict : {};
                let transitions = board.data.hasOwnProperty("transitionsDict") ? board.data.transitionsDict : {};
                let totalStates = [];

                for (const iotState of iotStates) {
                    // record this state's information
                    totalStates.push({
                        id: "node_" + iotState.idx,
                        time: iotState.time,
                        data: { label: "State " + iotState.idx }
                    });

                    // if it is a new state => create a new node for this state
                    if (!states.hasOwnProperty("node_" + iotState.idx)) {
                        let state = {
                            id: "node_" + iotState.idx,
                            type: "stateNode",
                            time: iotState.time,
                            position: { x: 50 + 300 * (parseInt(iotState.idx)), y: 100 },
                            positionAbsolute: { x: 50 + 300 * (parseInt(iotState.idx)), y: 100 },
                            data: { label: "State " + iotState.idx },
                            style: {
                                width: "150px",
                                height: "80px",
                                borderWidth: "1px",
                                borderStyle: "solid",
                                borderColor: "#6d8ee0",
                                padding: "10px",
                                borderRadius: "10px",
                                backgroundColor: "lightgrey",
                                display: "flex",
                                justifyContent: "center",
                                alignItems: "center"
                            },
                            zIndex: 1003
                        };
                        states[iotState.idx] = state;

                        if (iotState.prev_idx !== "-99") {
                            let transition = {
                                id: "edge_" + iotState.prev_idx + "-" + iotState.idx,
                                type: "transitionEdge",
                                source: "node_" + iotState.prev_idx,
                                target: "node_" + iotState.idx,
                                markerEnd: {
                                    type: MarkerType.ArrowClosed,
                                    width: 20,
                                    height: 20,
                                    color: '#FF0072',
                                },
                                data: {
                                    label: "action (" + iotState.prev_idx + "->" + iotState.idx + ")"
                                },
                                zIndex: 1003
                            };
                            transitions[transition.id] = transition;
                        }
                    }
                    //  if it is an existing state => just add an edge to the existing node of this state (but the time attribute is ignored in current method)
                    else {
                        if (!transitions.hasOwnProperty("edge_" + iotState.prev_idx + "-" + iotState.idx)) {
                            let transition = {
                                // current edge just means that there is a transition between, but not encode the temporal information (e.g. it's second time been here)
                                id: "edge_" + iotState.prev_idx + "-" + iotState.idx,
                                type: "transitionEdge",
                                source: "node_" + iotState.prev_idx,
                                target: "node_" + iotState.idx,
                                markerEnd: {
                                    type: MarkerType.ArrowClosed,
                                    width: 20,
                                    height: 20,
                                    color: '#FF0072',
                                },
                                data: {
                                    label: "action (" + iotState.prev_idx + "->" + iotState.idx + ")"
                                },
                                zIndex: 1003
                            }
                            transitions[transition.id] = transition;
                        }
                    }
                };

                boardData.statesDict = states;
                boardData.transitionsDict = transitions;
                setBoard((prevBoard) => ({ ...prevBoard, data: boardData }));
                setTotalStates(totalStates);
            })
    };

    const annotationSensing = () => {
        axios
            .get(window.BACKEND_ADDRESS + "/predict/" + board.title)
            .then((resp) => {
                let curState = resp.data;
                console.log("now iot state", curState);
                if (curState.length > 0) {
                    // Hint for User => current stage and current transition
                    let prevStateIdx = curState.length >= 3 ? curState[curState.length - 3].state : "-1";
                    let curStateIdx = curState.length >= 2 ? curState[curState.length - 2].state : "-1";
                    let nextStateIdx = curState[curState.length - 1].state;
                    addNewHighlight(curStateIdx, nextStateIdx);
                    deletePrevHightlight(prevStateIdx, curStateIdx);
                }
            })
    };

    return (
        <div className="board-div">
            <Helmet>
                <title>{board.title}</title>
            </Helmet>
            <MenuBar title={board.title} onSave={onSave} onTitleChange={handleTitleFocusOut} step={step} handleClickBack={handleClickBack} handleClickNext={handleClickNext} isSensing={isSensing} />
            <div className="main-board-div">
                <div className="top-side-div">
                    <h5>You are now at {stages[step]} Stage. {hints[step]}</h5>
                    {(() => {
                        switch (step) {
                            case 0:
                                return (<Button variant="contained" color={isSensing === -1 ? "primary" : "secondary"} onClick={handleClickExplore}>{isSensing === -1 ? "Explore" : "End Explore"}</Button>)
                            case 1:
                                return (<Button variant="contained" color={isSensing === -1 ? "primary" : "secondary"} onClick={handleClickAnnotate}>{isSensing === -1 ? "Start Annotation" : "End Annotation"}</Button>)
                            case 2:
                                return (<Button variant="contained">Verify</Button>)
                            default:
                                break;
                        }
                    })()}
                </div>
                <div className="mid-side-div">
                    {step === 0 &&
                        <NodeChart ref={nodeChartRef} step={step} states={states} setStates={setStates} transitions={transitions} setTransitions={setTransitions} />
                    }
                    {step === 1 &&
                        <div style={{ width: "100%", height: "100%" }}>
                            <div className="annotation-top-side-div">
                                <NodeChart ref={nodeChartRef} step={step} states={states} setStates={setStates} transitions={transitions} setTransitions={setTransitions} />
                            </div>
                            <div className="annotation-bottom-side-div">
                                {/* <TimelineChart totalStates={totalStates} /> */}
                                <SideNodeBar />
                            </div>
                        </div>
                    }
                </div>
            </div>
        </div>
    )
}