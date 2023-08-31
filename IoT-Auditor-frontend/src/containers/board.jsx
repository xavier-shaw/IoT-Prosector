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
import { stateNodeStyle } from "../shared/chartStyle";

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
    const nodeChartRef = useRef(null);
    const collagePanelRef = useRef(null);

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
            position = { x: prevNode.position.x, y: prevNode.position.y + 250 };
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
            style: stateNodeStyle,
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
    };

    const updateConfusionMatrix = () => {
        collagePanelRef.current.classifyStates();
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
                                        <NodeChart chart={chart} setChart={setChart} ref={nodeChartRef} step={step}
                                            updateConfusionMatrix={updateConfusionMatrix} setChartSelection={setChartSelection} />
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
                                        <NodeChart chart={chart} setChart={setChart} ref={nodeChartRef} step={step}
                                            updateConfusionMatrix={updateConfusionMatrix} setChartSelection={setChartSelection} />
                                    </Grid>
                                    <Grid item xs={5} className="panel-div" zeroMinWidth>
                                        <CollagePanel ref={collagePanelRef} board={board} chart={chart} chartSelection={chartSelection} />
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