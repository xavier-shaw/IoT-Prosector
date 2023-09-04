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
import { Button, Grid, Typography, Dialog, DialogActions, DialogTitle, } from "@mui/material";
import CheckIcon from '@mui/icons-material/Check';
import InstructionTable from "../components/InstructionTable";
import InteractionRecorder from "../components/InteractionRecorder";
import CollagePanel from "../components/CollagePanel";
import { childNodeoffsetY, nodeOffsetX, nodeOffsetY, semanticNodeMarginX, semanticNodeMarginY, semanticNodeOffsetX, stateNodeStyle } from "../shared/chartStyle";

export default function Board(props) {
    let params = useParams();
    const [board, setBoard] = useState({});
    const [step, setStep] = useState(0);
    const stages = ["Interaction", "Collage", "Verification"];
    const [isSensing, setIsSensing] = useState(-1);
    const [chart, setChart] = useState({});
    const [instructions, setInstructions] = useState([]);
    const [prevNode, setPrevNode] = useState(null);
    const [chartSelection, setChartSelection] = useState({ nodes: [], edges: [] });
    const [chainNum, setChainNum] = useState(0);
    const [status, setStatus] = useState("start");
    const [openDialog, setOpenDiaglog] = useState(false);

    const nodeChartRef = useRef(null);
    const collagePanelRef = useRef(null);
    const interactionRecorderRef = useRef(null);

    useEffect(() => {
        axios
            .get(window.BACKEND_ADDRESS + "/boards/" + params.boardId)
            .then((resp) => {
                let board_data = resp.data[0];
                setChart(board_data.chart);
                setInstructions(board_data.data.instructions);
                console.log("board data", board_data);
                setBoard(board_data);
            });
        return () => {
        }
    }, [params.boardId]);

    useEffect(() => {
        if (chart.hasOwnProperty("nodes")) {
            setChainNum(chart.nodes.filter((n) => n.data.status === "base state").length);
        }
    }, [chart])

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
        await axios
            .post(window.BACKEND_ADDRESS + "/boards/saveBoard", { boardId: board._id, updates: newBoard })
            .then((resp) => {
                console.log("successfully update board", resp.data);
            });
    };

    const addAction = (action) => {
        interactionRecorderRef.current.setAction(action);
        interactionRecorderRef.current.setOpenActionDialog(true);
    };

    const startCollage = async () => {
        await nodeChartRef.current.collageStates();
        setOpenDiaglog(true);
    }

    const endCollage = () => {
        nodeChartRef.current.updateAnnotation();
        setOpenDiaglog(false);
    }

    const updateMatrix = (nodes) => {
        collagePanelRef.current.classifyStates(nodes);
    }

    const createNode = (nodeIdx, status, state, action, edgeIdx) => {
        let newChart = { ...chart };
        let position;

        // create edge from prev node
        if (status === "state") {
            let newTransition = createEdge(edgeIdx, prevNode.id, nodeIdx, action);
            newChart.edges.push(newTransition);
            position = { x: prevNode.position.x, y: prevNode.position.y + nodeOffsetY };
        }
        else {
            // status = "base state" => base node
            position = { x: semanticNodeMarginX + nodeOffsetX * chainNum, y: semanticNodeMarginY };
            action = "base state action";
        }

        let newState = {
            id: nodeIdx,
            type: "stateNode",
            position: position,
            positionAbsolute: position,
            data: { label: state, status: status, action: action, prev: prevNode ? prevNode.id : null },
            style: stateNodeStyle,
            zIndex: 3
        };
        newChart.nodes.push(newState);
        setPrevNode(newState);
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
    };

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
                    {step === 1 &&
                        <>
                            <h6>&nbsp;First </h6>
                            <Button className="ms-2 me-2" size="small" color="primary" variant="contained" onClick={startCollage}>Collage</Button>
                            <h6>, then drag the states to debug.</h6>
                        </>
                    }
                    <Dialog open={openDialog} onClose={() => { setOpenDiaglog(false) }}>
                        <DialogTitle>Collage process completed.</DialogTitle>
                        <DialogActions>
                            <Button className='mt-2' variant="outlined" color="success" onClick={endCollage} startIcon={<CheckIcon />}>Confirm</Button>
                        </DialogActions>
                    </Dialog>
                </div>
                <Grid container columnSpacing={2} className="bottom-side-div">
                    <Grid item xs={7} className="panel-div">
                        <NodeChart board={board} chart={chart} setChart={setChart} ref={nodeChartRef} step={step}
                            setChartSelection={setChartSelection} updateMatrix={updateMatrix}/>
                    </Grid>
                    {(() => {
                        switch (step) {
                            case 0:
                                return (

                                    <Grid item xs={5} className="panel-div" zeroMinWidth>
                                        <div className="table-div">
                                            <InstructionTable instructions={instructions} setInstructions={setInstructions} addAction={addAction} status={status} />
                                        </div>
                                        <div className="action-div">
                                            <InteractionRecorder ref={interactionRecorderRef} board={board} chart={chart} createNode={createNode}
                                                chainNum={chainNum} setChainNum={setChainNum} status={status} setStatus={setStatus} />
                                        </div>
                                    </Grid>
                                );
                            case 1:
                                return (
                                    <Grid item xs={5} className="panel-div" zeroMinWidth>
                                        <CollagePanel ref={collagePanelRef} board={board} chart={chart} chartSelection={chartSelection} />
                                    </Grid>
                                );
                            case 2:
                                return (
                                    <Grid item xs={5} className="panel-div" zeroMinWidth>
                                        <div>

                                        </div>
                                    </Grid>
                                )
                            default:
                                break;
                        }
                    })()}
                </Grid>
            </div>
        </div>
    )
}