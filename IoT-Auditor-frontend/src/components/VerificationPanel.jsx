import React, { forwardRef, useImperativeHandle, useState, useEffect } from "react";
import OnlinePredictionIcon from '@mui/icons-material/OnlinePrediction';
import { Button, Dialog, DialogActions, DialogTitle, FormControl, InputLabel, LinearProgress, MenuItem, Select } from "@mui/material";
import "./VerificationPanel.css";
import axios from "axios";
import { v4 as uuidv4 } from "uuid";
import { colorPalette } from "../shared/chartStyle";

const VerificatopmPanel = forwardRef((props, ref) => {
    const { board, chart, status, setStatus, stateSequence, actionSequence, predictStates } = props;
    const [action, setAction] = useState("None");
    const [prediction, setPrediction] = useState(null);
    const [wrongPrediction, setWrongPrediction] = useState(false);
    const [correctState, setCorrectState] = useState("");
    const [verifyIdx, setVerifyIdx] = useState(-1);
    const graphWidth = 700;
    const graphHeight = 540;

    useImperativeHandle(ref, () => ({
        setAction,
        setDoingAction
    }));

    const startPrediction = () => {
        setVerifyIdx((prev) => (prev + 1));
        let stateId = stateSequence[verifyIdx + 1];
        let action = actionSequence[verifyIdx + 1];
        let predictionInfo = predictStates[stateId];
        setAction(action);
        setPrediction(predictionInfo);
        drawScatterplot(predictionInfo);
    };

    const onPredictionCorrect = () => {
        axios
            .get(window.HARDWARE_ADDRESS + "/verify", {
                params: {
                    device: board.title,
                    predict: prediction.predictState.data.representLabel,
                    correct: prediction.predictState.data.representLabel
                }
            })
            .then((resp) => {
                console.log(resp);
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
                    device: board.title,
                    predict: prediction.predictState.data.representLabel,
                    correct: correctState
                }
            })
            .then((resp) => {
                setWrongPrediction(false);
                setCorrectState("");
            })
    };

    const drawScatterplot = (prediction) => {
        let nodes = chart.nodes;
        let data = board.data.tsne_data_points_train;
        let labels = board.data.tsne_data_labels_train;
        let margin = 10;
        let cubeSize = 10;
        let legendMargin = 6;
        let graphOffsetX = 25;
        let graphOffsetY = 40;

        document.getElementById("graph-panel").innerHTML = "";
        let svg = d3.select("#graph-panel").append("svg")
            .attr("id", "svg")
            .attr("width", graphWidth)
            .attr("height", graphHeight);

        let parentNodes = nodes.filter((n) => !n.parentNode);

        const color = d3.scaleOrdinal()
            .domain(nodes.map(d => d.id))
            .range(colorPalette);

        let legend = svg.append("g")
            .attr("transform", `translate(${0}, ${margin})`)

        // Calculate the total width of legend items and labels
        const legendItems = legend
            .selectAll(".legend-item")
            .data(parentNodes)
            .enter()
            .append("g")
            .attr("class", "legend-item");

        legendItems
            .append("rect")
            .attr("fill", d => color(d.id))
            .attr("width", cubeSize)
            .attr("height", cubeSize)

        // legendItems
        //     .append("text")
        //     .text(d => d.data.representLabel)
        //     .style("font-size", 20)
        //     .style("font-family", "Times New Roman")
        //     .attr("transform", `translate(${cubeSize}, 10)`);

        // Calculate the total width of legend items and labels
        // Adjust the position of each label to prevent overlap
        let xOffset = 0;
        let yOffset = 0;
        const totalLegendWidth = legendItems.nodes().reduce((totalWidth, node) => {
            const bbox = node.querySelector('text').getBBox();
            d3.select(node).attr("transform", `translate(${xOffset}, ${yOffset})`)
            xOffset += bbox.width + cubeSize + legendMargin;
        }, 0);

        let xScaler = d3.scaleLinear()
            .domain([d3.min(data, d => d[0]), d3.max(data, d => d[0])])
            .range([graphOffsetX, graphWidth - graphOffsetX]);

        let yScaler = d3.scaleLinear()
            .domain([d3.min(data, d => d[1]), d3.max(data, d => d[1])])
            .range([graphOffsetY, graphHeight - graphOffsetY]);

        svg.append("g")
            .attr("transform", `translate(0,${yScaler(0)})`)
            .call(d3.axisBottom(xScaler))
            .selectAll(".tick text")
            .style("font-size", 20)
            .style("font-family", "Times New Roman")

        svg.append("g")
            .attr("transform", `translate(${xScaler(0)},0)`)
            .call(d3.axisLeft(yScaler))
            .selectAll(".tick text")
            .style("font-size", 20)
            .style("font-family", "Times New Roman")

        svg.append("g")
            .attr("fill", "none")
            .selectAll("circle")
            .data(data)
            .join("circle")
            .attr("fill", (d, i) => {
                let node = nodes.find((n) => n.id === labels[i])
                if (node.parentNode) {
                    return color(node.parentNode);
                }
                else {
                    return color(labels[i]);
                }
            })
            .attr("transform", d => `translate(${xScaler(d[0])},${yScaler(d[1])})`)
            .attr("r", 7);

        svg
            .append("circle")
            .attr("transform", `translate(${xScaler(prediction.data[0])}),${yScaler(prediction.data[1])}`)
            .attr("r", 7) // Adjust the radius as needed
            .style("fill", "black"); // You can set the fill color as desired
    };

    return (
        <div className="verification-panel-div">
            <h4>Verification</h4>
            <div className="graph-panel">
                <Skeleton className="m-auto" variant="rectangular" animation="wave" width={graphWidth} height={graphHeight} />
            </div>
            <div>
                {(() => {
                    switch (verifyIdx) {
                        case -1:
                            return (
                                <div>
                                </div>
                            )
                        default:
                            return (
                                <div>
                                    {prediction?.predictState?.data?.status !== "base state" && action !== "base state action" &&
                                        <div>
                                            <h4 style={{ fontFamily: "Times New Roman" }}>Previous Predicted State is: {prediction?.predictState?.data?.representLabel}</h4>
                                            <h4 style={{ fontFamily: "Times New Roman" }}>Your Action is: {action}</h4>
                                        </div>
                                    }
                                    <h4 style={{ fontFamily: "Times New Roman" }}>Current Predicted State is: {prediction?.predictState?.data?.representLabel}</h4>
                                </div>
                            )
                    }
                })()}

                <Button className="m-2" sx={{ fontWeight: "bold", fontSize: 20, fontFamily: "Times New Roman" }}
                    onClick={startPrediction} startIcon={<OnlinePredictionIcon />}>
                    {verifyIdx === -1? "Start State Prediction" : "Next State Prediction"}
                </Button>

                {
                    status === "verifying" &&
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
                                        sx={{ width: "200px" }}
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
            </div>
        </div >
    )
});

export default VerificatopmPanel;