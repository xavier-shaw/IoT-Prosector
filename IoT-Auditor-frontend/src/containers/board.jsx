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
import { Button, Grid, Typography, Dialog, DialogActions, DialogTitle, DialogContent, LinearProgress, } from "@mui/material";
import CheckIcon from '@mui/icons-material/Check';
import InstructionTable from "../components/InstructionTable";
import InteractionRecorder from "../components/InteractionRecorder";
import CollagePanel from "../components/CollagePanel";
import { childNodeoffsetY, edgeZIndex, nodeOffsetX, nodeOffsetY, semanticNodeMarginX, semanticNodeMarginY, semanticNodeOffsetX, stateNodeStyle, stateZIndex } from "../shared/chartStyle";
import { MarkerType } from "reactflow";

export default function Board(props) {
    let params = useParams();
    const [board, setBoard] = useState({});
    const [step, setStep] = useState(0);
    const stages = ["Interaction", "Collage", "Verification"];
    const [isSensing, setIsSensing] = useState(-1);
    const [chart, setChart] = useState({});
    const [instructions, setInstructions] = useState([]);
    const [prevNode, setPrevNode] = useState(null);
    const [chartSelection, setChartSelection] = useState(null);
    const [chainNum, setChainNum] = useState(0);
    const [status, setStatus] = useState("start");
    const [openDialog, setOpenDiaglog] = useState(false);
    const [waitForProcessing, setWaitForProcessing] = useState(false);
    const [finishProcess, setFinishProcess] = useState(false);
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

    const handleClickNext = () => {
        if (step === 0) {
            setWaitForProcessing(true);
            axios.get(window.HARDWARE_ADDRESS + "/waitForDataProcessing")
                .then((resp) => {
                    console.log(resp);
                    setFinishProcess(true);
                })
        }
        else {
            setStep((prevStep) => (prevStep + 1));
        }
    };

    const handleClickBack = () => {
        setStep((prevStep) => (prevStep - 1));
    };

    const toNextStage = () => {
        setWaitForProcessing(false);
        setFinishProcess(false);
        setStep((prev) => (prev + 1));
    };

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
    };

    const endCollage = () => {
        setOpenDiaglog(false);
    };

    const updateMatrix = (nodes) => {
        collagePanelRef.current.classifyStates(nodes);
    };

    const showHints = (node, type) => {
        if (type === "semantic") {
            nodeChartRef.current.showSemanticHints(node);
        }
        else if (type === "data") {
            nodeChartRef.current.showDataHints(node);
        }
    };

    const hideHints = (node, type) => {
        if (type === "semantic") {
            nodeChartRef.current.hideSemanticHints(node);
        }
        else if (type === "data") {
            nodeChartRef.current.hideDataHints(node);
        }
        else if (type === "all") {
            nodeChartRef.current.hideSemanticHints(node);
            nodeChartRef.current.hideDataHints(node);
        }
    };

    const previewFinalChart = () => {
        nodeChartRef.current.previewChart();
    };

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
            zIndex: stateZIndex
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
            originalSource: srcId,
            originalTarget: dstId,
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
            zIndex: edgeZIndex
        };

        return transition;
    };

    return (
        <div className="board-div">
            <Helmet>
                <title>{board.title}</title>
            </Helmet>
            <MenuBar title={board.title} onSave={onSave} onTitleChange={handleTitleFocusOut} step={step} handleClickBack={handleClickBack} handleClickNext={handleClickNext} />
            <div className="main-board-div">
                <div className="top-side-div">
                    <h6>You are now at the {stages[step]} Stage.</h6>
                    {step === 1 &&
                        <>
                            <h6>&nbsp;You can </h6>
                            <Button className="ms-2 me-2" size="small" color="primary" variant="contained" onClick={startCollage}>Collage</Button>
                            <h6>, interact, and .</h6>
                            <Button className="ms-2 me-2" size="small" color="primary" variant="contained" onClick={previewFinalChart}>Preview</Button>
                            <h6>the final state diagram.</h6>
                        </>
                    }
                    <Dialog open={openDialog}>
                        <DialogTitle>Collage process completed.</DialogTitle>
                        <DialogActions>
                            <Button className='mt-2' variant="outlined" color="success" onClick={endCollage} startIcon={<CheckIcon />}>Confirm</Button>
                        </DialogActions>
                    </Dialog>
                </div>
                <Grid container columnSpacing={2} className="bottom-side-div">
                    <Grid item xs={7} className="panel-div">
                        <NodeChart board={board} chart={chart} setChart={setChart} ref={nodeChartRef} step={step}
                            chartSelection={chartSelection} setChartSelection={setChartSelection} updateMatrix={updateMatrix} />
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
                                        <CollagePanel ref={collagePanelRef} board={board} chart={chart} chartSelection={chartSelection}
                                            showHints={showHints} hideHints={hideHints} />
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

            <Dialog open={waitForProcessing}>
                <DialogTitle>Please wait until the data analysis is done.</DialogTitle>
                <DialogContent>
                    {!finishProcess && <div>
                        <p>It takes up to 30 seconds to finish.</p>
                        <LinearProgress />
                    </div>}
                    {finishProcess && <h5>The data analysis is finished.</h5>}
                </DialogContent>
                <DialogActions>
                    <Button variant="contained" color="primary" disabled={!finishProcess} onClick={toNextStage}>To Collage Stage</Button>
                </DialogActions>
            </Dialog>
        </div>
    )
}