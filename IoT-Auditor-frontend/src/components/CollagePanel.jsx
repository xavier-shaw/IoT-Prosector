import React, { forwardRef, useEffect, useImperativeHandle, useState } from "react";
import * as d3 from "d3";
import "./CollagePanel.css";
import { Button, Grid, Skeleton } from "@mui/material";
import axios from "axios";

const CollagePanel = forwardRef((props, ref) => {
    let { board, chart, chartSelection, showHints, hideHints } = props;
    const [selectedNode, setSelectedNode] = useState(null);
    const [selectedNodeVideo, setSelectedNodeVideo] = useState(null);
    const [prevNode, setPrevNode] = useState(null);
    const [prevNodeVideo, setPrevNodeVideo] = useState(null);
    const [actionVideo, setActionVideo] = useState(null);
    const [figure, setFigure] = useState("confusion matrix");
    const [matrixData, setMatrixData] = useState({});
    const [hint, setHint] = useState("");
    const graphWidth = 670;
    const graphHeight = 510;
    const videoWidth = 200;
    const videoHeight = 200;

    useImperativeHandle(ref, () => ({
        classifyStates,
    }));

    useEffect(() => {
        if (chartSelection.type === "stateNode") {
            if (selectedNode) {
                hideHints(selectedNode, "all");
                setHint("");
            }
            let node_id = chartSelection.id;
            let prev_id = chartSelection.data.prev;
            let curNode = chart.nodes.find((e) => e.id === node_id);
            setSelectedNode(curNode);
            setVideoById(node_id, setSelectedNodeVideo);
            if (prev_id) {
                let prevNode = chart.nodes.find((e) => e.id === prev_id);
                setPrevNode(prevNode);
                setVideoById(prev_id, setPrevNodeVideo);
                let edge_id = chart.edges.find((e) => e.source === prev_id && e.target === node_id).id;
                setVideoById(edge_id, setActionVideo);
            }
            else {
                setPrevNode(null);
                setPrevNodeVideo(null);
                setActionVideo(null);
            }
        }
    }, [chartSelection]);

    const setVideoById = (id, setStateFunction) => {
        axios
            .get(window.BACKEND_ADDRESS + "/video/get/" + id)
            .then((resp) => {
                setStateFunction(resp.data[0].video);
            })
    }

    const classifyStates = (nodes) => {
        axios
            .post(window.HARDWARE_ADDRESS + "/classification", {
                device: board.title,
                nodes: nodes
            })
            .then((resp) => {
                let matrixData = {
                    accuracy: resp.data.accuracy,
                    matrix: resp.data.confusionMatrix,
                    states: resp.data.states
                }

                drawConfusionMatrix(matrixData);
                setMatrixData(matrixData);
                console.log("update matrix: ", acc);
            })
    };

    const drawConfusionMatrix = (matrixData) => {
        let accuracyOffset = 30;
        let matrixOffset = 10;
        let labelOffset = 5;
        let labelRotate = -20;
        let legendOffsetX = 20;
        let legendOffsetY = 0;
        let legendWidth = 15;
        let margin = 80;
        let offsetX = 150;
        let offsetY = accuracyOffset + matrixOffset;
        let legendHeight = graphHeight - offsetY - margin;

        document.getElementById("confusion-matrix").innerHTML = "";
        let svg = d3.select("#confusion-matrix").append("svg")
            .attr("width", graphWidth)
            .attr("height", graphHeight);

        svg.append("text")
            .attr("x", graphWidth / 2)
            .attr("y", accuracyOffset)
            .attr("font-size", 20)
            .style("text-anchor", "middle")
            .text("Average Accuracy: " + matrixData.accuracy);

        // svg.append("text")
        //     .attr("x", 15)
        //     .attr("y", offsetY + legendHeight / 2)
        //     .attr("font-size", 20)
        //     .style("text-anchor", "middle")
        //     .text("True Labels")
        //     .style("writing-mode", "tb")
        //     .style("glyph-orientation-vertical", 90)

        let cellSize = legendHeight / matrixData.matrix.length;
        // let colorScale = d3.scaleSequential(d3.interpolateRgbBasis(["#FFFFDD", "#3E9583", "#1F2D86"]))
        //     .domain([0, 1])
        let colorScale = d3.scaleLinear()
            .domain([0, 0.5, 1])
            .range(["#FFFFDD", "#3E9583", "#1F2D86"])

        svg.selectAll('rect')
            .data(matrixData.matrix)
            .enter().append('g')
            .attr("transform", function (d, i) {
                return "translate(" + offsetX + ", " + (i * cellSize + offsetY) + ")"
            })
            .selectAll('rect')
            .data(d => d)
            .enter().append('rect')
            .attr('x', (d, i) => i * cellSize)
            .attr('y', (d, i) => 0)
            .attr('width', cellSize)
            .attr('height', cellSize)
            .attr('fill', (d) => colorScale(d))
            .attr('stroke', 'black')

        svg.selectAll('.states-text')
            .data(matrixData.matrix)
            .enter().append('g')
            .attr("transform", function (d, i) {
                return "translate(" + offsetX + ", " + (i * cellSize + offsetY) + ")"
            })
            .selectAll('.state-text')
            .data(d => d)
            .enter().append('text')
            .attr('x', (d, i) => i * cellSize + cellSize / 2)
            .attr('y', (d, i) => cellSize / 2)
            .attr('text-anchor', 'middle')
            .attr('font-size', 12)
            .attr('dy', '.35em')
            .text(d => d);

        svg.selectAll('.true-states')
            .data(matrixData.states)
            .enter()
            .append('text')
            .text((d) => d)
            .attr("x", offsetX - labelOffset)
            .attr("y", (d, i) => offsetY + i * cellSize + cellSize / 2)
            .attr("alignment-baseline", "middle")
            .attr("text-anchor", "end")
            .attr("font-size", 16)

        svg.selectAll('.predict-states')
            .data(matrixData.states)
            .enter()
            .append('text')
            .text((d) => d)
            .attr("x", (d, i) => offsetX + i * cellSize + cellSize / 2)
            .attr("y", offsetY + legendHeight + labelOffset + 10)
            .attr("text-anchor", "end")
            .attr("font-size", 16)
            .attr("transform", (d, i) => "rotate(" + labelRotate + " " + (offsetX + i * cellSize + cellSize / 2) + " " + (offsetY + legendHeight + labelOffset + 10) + ")")


        //Calculate the variables for gradient
        var gradientScale = d3.scaleLinear()
            .domain([0, 1])
            .range([0, legendHeight])
        var numStops = 10;
        var gradientPoints = [];
        for (var i = 0; i < numStops; i++) {
            gradientPoints.push(i / (numStops - 1));
        }

        //Create the gradient
        svg.append("defs")
            .append("linearGradient")
            .attr("id", "legend-traffic")
            .attr("x1", "0%").attr("y1", "100%")
            .attr("x2", "0%").attr("y2", "0%")
            .selectAll("stop")
            .data(d3.range(numStops))
            .enter().append("stop")
            .attr("offset", function (d, i) {
                return gradientScale(gradientPoints[i]) / legendHeight;
            })
            .attr("stop-color", function (d, i) {
                return colorScale(gradientPoints[i]);
            });

        var legendsvg = svg.append("g")
            .attr("transform", "translate(" + (offsetX + legendHeight + legendOffsetX) + "," + (offsetY + legendOffsetY) + ")");

        //Draw the Rectangle
        legendsvg.append("rect")
            .attr("x", 0)
            .attr("y", 0)
            .attr("width", legendWidth)
            .attr("height", legendHeight)
            .style("fill", "url(#legend-traffic)");

        //Set scale for x-axis
        var xScale = d3.scaleLinear()
            .range([0, legendHeight])
            .domain([1, 0]);

        //Define x-axis
        var xAxis = d3.axisRight(xScale)
            .ticks(5)


        legendsvg.append("g")
            .attr("transform", "translate(" + legendWidth + ",0)")
            .call(xAxis);
    };

    const handleClickSelect = (type) => {
        setFigure(type);
        if (type === "confusion matrix") {
            drawConfusionMatrix(matrixData);
        }
        else if (type === "distribution") {
            console.log()
        }
    };

    const handleShowSemanticHints = () => {
        if (hint === "semantic") {
            setHint("");
            hideHints(selectedNode, "semantic");
        }
        else {
            setHint("semantic");
            showHints(selectedNode, "semantic");
        }
    };

    const handleShowDataHints = () => {
        if (hint === "data") {
            setHint("");
            hideHints(selectedNode, "data");
        }
        else {
            setHint("data");
            showHints(selectedNode, "data");
        }
    };

    return (
        <div className="collage-panel-div">
            <div className="select-panel">
                <Button variant={figure === "confusion matrix" ? "contained" : "outlined"} color="primary" onClick={() => handleClickSelect("confusion matrix")}>
                    Confusion Matrix
                </Button>
                <Button variant={figure === "distribution" ? "contained" : "outlined"} color="primary" onClick={() => handleClickSelect("distribution")}>
                    Distribution
                </Button>
            </div>
            <div id="confusion-matrix">
                <Skeleton className="m-auto" variant="rectangular" animation="wave" width={600} height={550} />
            </div>

            {/* TODO: 1. Stepwise Debugging => State Info (video + timewise-next-state + related action / labeled action) 
                      2. Group Node and Split Node => change should be demonstrated by confusion matrix  
            */}
            <h4>State Information</h4>
            <div className="d-flex justify-content-evenly">
                <Button variant={hint === "semantic" ? "contained" : "outlined"} color={hint === "semantic" ? "success" : "primary"}
                    disabled={hint === "data" || !selectedNode} size="small" onClick={handleShowSemanticHints}>
                    Semantic Hint
                </Button>
                <Button variant={hint === "data" ? "contained" : "outlined"} color={hint === "data" ? "success" : "primary"}
                    disabled={hint === "semantic" || !selectedNode} size="small" onClick={handleShowDataHints}>
                    Data Hint
                </Button>
            </div>
            <div className="state-info-div">
                {prevNode && prevNodeVideo &&
                    <div>
                        <h5>Previous State</h5>
                        <h6>{prevNode.data.label}</h6>
                        <video src={prevNodeVideo} controls width={videoWidth} height={videoHeight} />
                    </div>
                }
                {prevNode && actionVideo &&
                    <div>
                        <h5>Action</h5>
                        <h6>{selectedNode.data.action}</h6>
                        <video src={actionVideo} controls width={videoWidth} height={videoHeight} />
                    </div>
                }
                {!prevNode && selectedNode &&
                    <h5>No previous state or action</h5>
                }
                {!selectedNode &&
                    <h5>Please select a state or action</h5>
                }
                {selectedNode && selectedNodeVideo &&
                    <div>
                        <h5>Current State</h5>
                        <h6>{selectedNode.data.label}</h6>
                        <video src={selectedNodeVideo} controls width={videoWidth} height={videoHeight} />
                    </div>
                }
            </div>
            {(selectedNode === null || selectedNodeVideo === null) &&
                <Grid container columnSpacing={1}>
                    <Grid item xs={4}>
                        {/* <Skeleton variant="h6" animation={false}/> */}
                        <Skeleton className="m-auto" variant="rectangular" animation={false} width={videoWidth} height={videoHeight} />
                    </Grid>
                    <Grid item xs={4}>
                        <Skeleton className="m-auto" variant="rectangular" animation={false} width={videoWidth} height={videoHeight} />
                    </Grid>
                    <Grid item xs={4}>
                        <Skeleton className="m-auto" variant="rectangular" animation={false} width={videoWidth} height={videoHeight} />
                    </Grid>
                </Grid>
            }
        </div>
    )
})

export default CollagePanel;