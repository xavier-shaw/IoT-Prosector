import React, { useCallback, useRef } from "react";
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
import CollagePanel from "../components/CollagePanel";

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
    const [chart, setChart] = useState({});
    const [instructions, setInstructions] = useState([]);
    const [prevNode, setPrevNode] = useState(null);
    const [chartSelection, setChartSelection] = useState({ nodes: [], edges: [] });
    const nodeChartRef = useRef(null);

    useEffect(() => {
        axios
            .get(window.BACKEND_ADDRESS + "/boards/" + params.boardId)
            .then((resp) => {
                let board_data = resp.data[0];
                // board_data.data = JSON.parse(board_data.data);
                // board_data.chart = JSON.parse(board_data.chart);
                setChart(board_data.chart);
                setInstructions(board_data.data.instructions);
                console.log("board data", board_data);
                setBoard(board_data);
                // axios
                //     .get(window.BACKEND_ADDRESS + "/shared/update/" + board_data.title)
                //     .then((resp) => {
                //         console.log("device", resp.data);
                //     })
            });
        return () => {
        }
    }, [params.boardId]);

    const handleClickNext = async () => {
        await onSave();
        setStep((prevStep) => (prevStep + 1));
    };

    const handleClickBack = () => {
        setStep((prevStep) => (prevStep - 1));
    }

    const handleTitleFocusOut = (titleText) => {
        console.log("Update board title:", titleText);
        setBoard(prevBoard => ({ ...prevBoard, title: titleText }));
    };

    const onSave = async () => {
        let newBoard = { ...board };
        let newChart = nodeChartRef.current.updateAnnotation();
        let newInstructions = instructions; 
        newBoard.chart = newChart;
        newBoard.data.instructions = newInstructions;
        setBoard(newBoard);
        console.log("ready to update board", newBoard);
        axios
            .post(window.BACKEND_ADDRESS + "/boards/saveBoard", { boardId: board._id, updates: newBoard })
            .then((resp) => {
                console.log("successfully update board", resp.data);
            });
    };

    const createNode = (nodeIdx, status, action, edgeIdx) => {
        let newChart = { ...chart };
        let index = newChart.nodes.length;
        let position;

        // create edge from prev node
        if (status === "Action") {
            let newTransition = createEdge(edgeIdx, prevNode.id, nodeIdx, action);
            newChart.edges.push(newTransition);
            position = { x: prevNode.position.x, y: prevNode.position.y + 200 };
            if (action === "") {
                action = "Action";
            }
        }
        else {
            // base node
            let baseNodeCnt = (newChart.nodes.filter((n) => n.data.status === "Base")).length;
            position = { x: 10 + 200 * baseNodeCnt, y: 10 };
            action = "Base";
        }

        let state = {
            id: nodeIdx,
            type: "stateNode",
            position: position,
            positionAbsolute: position,
            data: { label: "State " + index, status: status, action: action, prev: prevNode ? prevNode.id : null },
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
        newChart.nodes.push(state);
        setPrevNode(state);
        setChart(newChart);
    };

    const createEdge = (edgeIdx, srcId, dstId, action) => {
        let transition = {
            id: edgeIdx,
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

        return transition;
    }

    // const deletePrevHightlight = (prevStateIdx, curStateIdx) => {
    //     let boardChart = board.chart;
    //     boardChart.nodes = boardChart.nodes.map((node) => {
    //         if (node.id === "node_" + curStateIdx) {
    //             node.style = { ...node.style, backgroundColor: "#788bff" };
    //             let prevTransitionLabel = document.getElementById("edge_" + prevStateIdx + "-" + curStateIdx + "_label");
    //             prevTransitionLabel.style.backgroundColor = "#f4a261";
    //         };

    //         return node;
    //     });

    //     setBoard((prevBoard) => ({ ...prevBoard, chart: boardChart }));
    // };

    // const addNewHighlight = (curStateIdx, nextStateIdx) => {
    //     let boardChart = board.chart;
    //     boardChart.nodes = boardChart.nodes.map((node) => {
    //         if (node.id === "node_" + nextStateIdx) {
    //             node.style = { ...node.style, backgroundColor: "skyblue" };
    //             let nextTransitionEdge = document.getElementById("edge_" + curStateIdx + "-" + nextStateIdx + "_label");
    //             nextTransitionEdge.style.backgroundColor = "skyblue";
    //         };

    //         return node;
    //     });

    //     setBoard((prevBoard) => ({ ...prevBoard, chart: boardChart }));
    // };

    // const sensing = () => {
    //     axios.
    //         get(window.BACKEND_ADDRESS + "/states/" + board.title)
    //         .then((resp) => {
    //             console.log("iot states", resp.data);
    //             let iotStates = resp.data;
    //             let boardData = board.data;
    //             let boardChart = board.chart;
    //             let newStatesDict = { ...statesDict };
    //             let newTransitionsDict = { ...transitionsDict };

    //             for (const iotState of iotStates) {
    //                 // if it is a new state => create a new node for this state
    //                 if (!statesDict.hasOwnProperty("node_" + iotState.idx)) {
    //                     let state = {
    //                         id: "node_" + iotState.idx,
    //                         type: "stateNode",
    //                         time: iotState.time,
    //                         position: { x: 50 + 400 * (parseInt(iotState.idx) + 1), y: 100 },
    //                         positionAbsolute: { x: 50 + 400 * (parseInt(iotState.idx) + 1), y: 100 },
    //                         data: { label: "State " + iotState.idx },
    //                         style: {
    //                             width: "150px",
    //                             height: "80px",
    //                             borderWidth: "1px",
    //                             borderStyle: "solid",
    //                             padding: "10px",
    //                             borderRadius: "10px",
    //                             backgroundColor: "#788bff",
    //                             display: "flex",
    //                             justifyContent: "center",
    //                             alignItems: "center"
    //                         },
    //                         zIndex: 3
    //                     };
    //                     newStatesDict[state.id] = state;
    //                     boardChart.nodes.push(state);

    //                     if (iotState.prev_idx !== "-99") {
    //                         let transition = {
    //                             id: "edge_" + iotState.prev_idx + "-" + iotState.idx,
    //                             type: "transitionEdge",
    //                             source: "node_" + iotState.prev_idx,
    //                             target: "node_" + iotState.idx,
    //                             markerEnd: {
    //                                 type: MarkerType.ArrowClosed,
    //                                 width: 30,
    //                                 height: 30,
    //                                 color: '#FF0072',
    //                             },
    //                             style: {
    //                                 strokeWidth: 2,
    //                                 stroke: '#000000',
    //                             },
    //                             data: {
    //                                 label: "action (" + iotState.prev_idx + "->" + iotState.idx + ")"
    //                             },
    //                             zIndex: 4
    //                         };
    //                         newTransitionsDict[transition.id] = transition;
    //                         boardChart.edges.push(transition);
    //                     }
    //                 }
    //                 //  if it is an existing state => just add an edge to the existing node of this state (but the time attribute is ignored in current method)
    //                 else {
    //                     if (!transitionsDict.hasOwnProperty("edge_" + iotState.prev_idx + "-" + iotState.idx)) {
    //                         let transition = {
    //                             // current edge just means that there is a transition between, but not encode the temporal information (e.g. it's second time been here)
    //                             id: "edge_" + iotState.prev_idx + "-" + iotState.idx,
    //                             type: "transitionEdge",
    //                             source: "node_" + iotState.prev_idx,
    //                             target: "node_" + iotState.idx,
    //                             markerEnd: {
    //                                 type: MarkerType.ArrowClosed,
    //                                 width: 30,
    //                                 height: 30,
    //                                 color: '#FF0072',
    //                             },
    //                             style: {
    //                                 strokeWidth: 2,
    //                                 stroke: '#000000',
    //                             },
    //                             data: {
    //                                 label: "action (" + iotState.prev_idx + "->" + iotState.idx + ")"
    //                             },
    //                             zIndex: 4
    //                         }
    //                         transitionsDict[transition.id] = transition;
    //                         boardChart.edges.push(transition);
    //                     }
    //                 }
    //             };

    //             boardData.statesDict = newStatesDict;
    //             boardData.transitionsDict = newTransitionsDict;
    //             setBoard((prevBoard) => ({ ...prevBoard, data: boardData, chart: boardChart }));
    //         })
    // };

    // const annotationSensing = () => {
    //     axios
    //         .get(window.BACKEND_ADDRESS + "/predict/" + board.title)
    //         .then((resp) => {
    //             let curState = resp.data;
    //             console.log("now iot state", curState);
    //             if (curState.length > 0) {
    //                 // Hint for User => current stage and current transition
    //                 let curStateIdx = curState.length >= 2 ? curState[curState.length - 2].state : "-1";
    //                 let nextStateIdx = curState[curState.length - 1].state;
    //                 addNewHighlight(curStateIdx, nextStateIdx);
    //                 setTimeout(() => {
    //                     deletePrevHightlight(curStateIdx, nextStateIdx);
    //                 }, 900);
    //             }
    //         })
    // };

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
                {(() => {
                    switch (step) {
                        case 0:
                            return (
                                <Grid container columnSpacing={2} className="bottom-side-div">
                                    <Grid item xs={3} className="table-div">
                                        <InstructionTable instructions={instructions} setInstructions={setInstructions} />
                                    </Grid>
                                    <Grid item xs={6} className="panel-div">
                                        <NodeChart chart={chart} setChart={setChart}
                                            ref={nodeChartRef} step={step} setChartSelection={setChartSelection} />
                                    </Grid>
                                    <Grid item xs={3} className="panel-div" zeroMinWidth>
                                        <InteractionRecorder board={board} createNode={createNode} />
                                    </Grid>
                                </Grid>
                            );
                        case 1:
                            return (
                                <Grid container columnSpacing={2} className="bottom-side-div">
                                    <Grid item xs={7} className="panel-div">
                                        {/* <NodeChart board={board} ref={nodeChartRef} step={step} /> */}
                                        <div style={{ width: "100%", height: "100%" }}>
                                            <div className="panel-div">
                                                <NodeChart chart={chart} setChart={setChart}
                                                    ref={nodeChartRef} step={step} setChartSelection={setChartSelection} />
                                            </div>
                                            {/* <div className="collage-bottom-side-div">
                                                <SideNodeBar />
                                            </div> */}
                                        </div>
                                    </Grid>
                                    <Grid item xs={5} className="panel-div" zeroMinWidth>
                                        <CollagePanel board={board} chart={chart} chartSelection={chartSelection} />
                                    </Grid>
                                </Grid>
                            )
                        default:
                            break;
                    }
                })()}

            </div>
        </div>
    )
}