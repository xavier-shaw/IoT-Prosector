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
import { Button, Grid, Typography } from "@mui/material";
import SideNodeBar from "../components/SideNodeBar";
import InstructionTable from "../components/InstructionTable";
import InteractionRecorder from "../components/InteractionRecorder";
import { transition } from "d3";

export default function Board(props) {
    let params = useParams();
    const [board, setBoard] = useState({});
    const [step, setStep] = useState(0);
    const stages = ["Interaction", "Collage", "Verification"];
    // const hints = [
    //     "Click button to start or stop generating state diagram.",
    //     "Click a node or edge to annotate, then confirm the annotation.",
    //     "Click button to begin your verification."
    // ]
    const [isSensing, setIsSensing] = useState(-1);
    const [statesDict, setStatesDict] = useState({});
    const [transitionsDict, setTransitionsdict] = useState({});
    const [instructions, setInstructions] = useState([]);
    const nodeChartRef = useRef(null);

    useEffect(() => {
        axios
            .get(window.BACKEND_ADDRESS + "/boards/" + params.boardId)
            .then((resp) => {
                let board_data = resp.data[0];
                board_data.data = JSON.parse(board_data.data);
                board_data.chart = JSON.parse(board_data.chart);
                if (board_data.data.hasOwnProperty("instructions")) {
                    setInstructions(board_data.data.instructions);
                };
                if (board_data.data.hasOwnProperty("statesDict")) {
                    setStatesDict(board_data.data.statesDict);
                };
                if (board_data.data.hasOwnProperty("transitionsDict")) {
                    setTransitionsdict(board_data.data.transitionsDict);
                };
                console.log("board data", board_data);
                setBoard(board_data);
                axios
                    .get(window.BACKEND_ADDRESS + "/shared/update/" + board_data.title)
                    .then((resp) => {
                        console.log("device", resp.data);
                    })
            });
        return () => {
        }
    }, [params.boardId]);

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

    const onSave = () => {
        let newBoard = {...board};
        let boardChart = nodeChartRef.current.updateAnnotation();
        newBoard.chart = boardChart;

        let newStatesDict = {...statesDict};
        let newTransitionDict = {...transitionsDict};
        for (const state of boardChart.nodes) {
            let state_id = state["id"];
            statesDict[state_id] = state;
        };
        for (const transition of boardChart.edges) {
            let transition_id = transition["id"];
            transitionsDict[transition_id] = transition;
        };
        setStatesDict(newStatesDict);
        setTransitionsdict(newTransitionDict);
        newBoard.data.statesDict = newStatesDict;
        newBoard.data.transitionsDict = newTransitionDict;
        newBoard.data.instructions = instructions;
        setBoard(newBoard);
        newBoard.data= JSON.stringify(newBoard.data);
        newBoard.chart = JSON.stringify(newBoard.chart);
        axios
            .post(window.BACKEND_ADDRESS + "/boards/saveBoard", { boardId: board._id, updates: newBoard })
            .then((resp) => {
                console.log("update board", resp.data);
            });
    };

    const createNode = (node_idx, status, action) => {
        let newStatesDict = {...statesDict};
        let boardChart = {...board.chart};
        let index = newStatesDict.length;
        let position;
        // create edge from last node
        if (status === "Action") {
            let lastNode = newStatesDict[index - 1];
            let newTransition = createEdge(lastNode.id, node_idx, action);
            boardChart.edges.push(newTransition);
            position = { x: lastNode.position.x, y: lastNode.position.y + 100};
        }
        else {
            // base node
            let baseNodeCnt = (boardChart.nodes.filter((n) => n.data.status === "Base")).length;
            position = { x: 10 + 200 * baseNodeCnt, y: 10};
        }

        let state = {
            id: node_idx,
            type: "stateNode",
            position: position,
            positionAbsolute: position,
            data: { label: "State " + index, status: status },
            style: {
                width: "150px",
                height: "80px",
                borderWidth: "1px",
                borderStyle: "solid",
                padding: "10px",
                borderRadius: "10px",
                backgroundColor: "#788bff",
                display: "flex",
                justifyContent: "center",
                alignItems: "center"
            },
            zIndex: 3
        };
        boardChart.nodes.push(state);
        newStatesDict[node_idx] = state;

        setBoard((prevBoard) => ({...prevBoard, chart: boardChart}));
        setStatesDict(newStatesDict);
    };

    const createEdge = (srcId, dstId, action) => {
        let newTransitionsDict = {...transitionsDict};
        let transition = {
            id: srcId + "-" + dstId,
            type: "transitionEdge",
            source: srcId,
            target: dstId,
            markerEnd: {
                type: MarkerType.ArrowClosed,
                width: 30,
                height: 30,
                color: '#FF0072',
            },
            style: {
                strokeWidth: 2,
                stroke: '#000000',
            },
            data: {
                label: action
            },
            zIndex: 4
        };

        newTransitionsDict[transition.id] = transition;
        setTransitionsdict(newTransitionsDict);
        return transition;
    }

    const deletePrevHightlight = (prevStateIdx, curStateIdx) => {
        let boardChart = board.chart;
        boardChart.nodes = boardChart.nodes.map((node) => {
            if (node.id === "node_" + curStateIdx) {
                node.style = { ...node.style, backgroundColor: "#788bff" };
                let prevTransitionLabel = document.getElementById("edge_" + prevStateIdx + "-" + curStateIdx + "_label");
                prevTransitionLabel.style.backgroundColor = "#f4a261";
            };

            return node;
        });

        setBoard((prevBoard) => ({ ...prevBoard, chart: boardChart }));
    };

    const addNewHighlight = (curStateIdx, nextStateIdx) => {
        let boardChart = board.chart;
        boardChart.nodes = boardChart.nodes.map((node) => {
            if (node.id === "node_" + nextStateIdx) {
                node.style = { ...node.style, backgroundColor: "skyblue" };
                let nextTransitionEdge = document.getElementById("edge_" + curStateIdx + "-" + nextStateIdx + "_label");
                nextTransitionEdge.style.backgroundColor = "skyblue";
            };

            return node;
        });

        setBoard((prevBoard) => ({ ...prevBoard, chart: boardChart }));
    };

    const sensing = () => {
        axios.
            get(window.BACKEND_ADDRESS + "/states/" + board.title)
            .then((resp) => {
                console.log("iot states", resp.data);
                let iotStates = resp.data;
                let boardData = board.data;
                let boardChart = board.chart;
                let newStatesDict = {...statesDict};
                let newTransitionsDict = {...transitionsDict};

                for (const iotState of iotStates) {
                    // if it is a new state => create a new node for this state
                    if (!statesDict.hasOwnProperty("node_" + iotState.idx)) {
                        let state = {
                            id: "node_" + iotState.idx,
                            type: "stateNode",
                            time: iotState.time,
                            position: { x: 50 + 400 * (parseInt(iotState.idx) + 1), y: 100 },
                            positionAbsolute: { x: 50 + 400 * (parseInt(iotState.idx) + 1), y: 100 },
                            data: { label: "State " + iotState.idx },
                            style: {
                                width: "150px",
                                height: "80px",
                                borderWidth: "1px",
                                borderStyle: "solid",
                                padding: "10px",
                                borderRadius: "10px",
                                backgroundColor: "#788bff",
                                display: "flex",
                                justifyContent: "center",
                                alignItems: "center"
                            },
                            zIndex: 3
                        };
                        newStatesDict[state.id] = state;
                        boardChart.nodes.push(state);

                        if (iotState.prev_idx !== "-99") {
                            let transition = {
                                id: "edge_" + iotState.prev_idx + "-" + iotState.idx,
                                type: "transitionEdge",
                                source: "node_" + iotState.prev_idx,
                                target: "node_" + iotState.idx,
                                markerEnd: {
                                    type: MarkerType.ArrowClosed,
                                    width: 30,
                                    height: 30,
                                    color: '#FF0072',
                                },
                                style: {
                                    strokeWidth: 2,
                                    stroke: '#000000',
                                },
                                data: {
                                    label: "action (" + iotState.prev_idx + "->" + iotState.idx + ")"
                                },
                                zIndex: 4
                            };
                            newTransitionsDict[transition.id] = transition;
                            boardChart.edges.push(transition);
                        }
                    }
                    //  if it is an existing state => just add an edge to the existing node of this state (but the time attribute is ignored in current method)
                    else {
                        if (!transitionsDict.hasOwnProperty("edge_" + iotState.prev_idx + "-" + iotState.idx)) {
                            let transition = {
                                // current edge just means that there is a transition between, but not encode the temporal information (e.g. it's second time been here)
                                id: "edge_" + iotState.prev_idx + "-" + iotState.idx,
                                type: "transitionEdge",
                                source: "node_" + iotState.prev_idx,
                                target: "node_" + iotState.idx,
                                markerEnd: {
                                    type: MarkerType.ArrowClosed,
                                    width: 30,
                                    height: 30,
                                    color: '#FF0072',
                                },
                                style: {
                                    strokeWidth: 2,
                                    stroke: '#000000',
                                },
                                data: {
                                    label: "action (" + iotState.prev_idx + "->" + iotState.idx + ")"
                                },
                                zIndex: 4
                            }
                            transitionsDict[transition.id] = transition;
                            boardChart.edges.push(transition);
                        }
                    }
                };

                boardData.statesDict = newStatesDict;
                boardData.transitionsDict = newTransitionsDict;
                setBoard((prevBoard) => ({ ...prevBoard, data: boardData, chart: boardChart }));
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
                    let curStateIdx = curState.length >= 2 ? curState[curState.length - 2].state : "-1";
                    let nextStateIdx = curState[curState.length - 1].state;
                    addNewHighlight(curStateIdx, nextStateIdx);
                    setTimeout(() => {
                        deletePrevHightlight(curStateIdx, nextStateIdx);
                    }, 900);
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
                    <h6>You are now at the {stages[step]} Stage.</h6>
                    {/* <h5>You are now at {stages[step]} Stage. {hints[step]}</h5> */}
                    {/* {(() => {
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
                    })()} */}
                </div>
                <Grid container columnSpacing={2} className="bottom-side-div">
                    <Grid item xs={3} className="left-side-div">
                        <InstructionTable instructions={instructions} setInstructions={setInstructions} />
                    </Grid>
                    <Grid item xs={6} className="mid-side-div">
                        {step === 0 &&
                            <NodeChart board={board} ref={nodeChartRef} step={step} />
                        }
                        {step === 1 &&
                            <div style={{ width: "100%", height: "100%" }}>
                                <div className="annotation-top-side-div">
                                    <NodeChart board={board} ref={nodeChartRef} step={step} />
                                </div>
                                <div className="annotation-bottom-side-div">
                                    {/* <TimelineChart totalStates={totalStates} /> */}
                                    <SideNodeBar />
                                </div>
                            </div>
                        }
                    </Grid>
                    <Grid item xs={3} className="right-side-div" zeroMinWidth>
                        <InteractionRecorder createNode={createNode}/>
                    </Grid>
                </Grid>
            </div>
        </div>
    )
}