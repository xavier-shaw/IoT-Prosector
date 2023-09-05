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
    const [classificationData, setClassificationData] = useState({});
    const [hint, setHint] = useState("");
    const graphWidth = 670;
    const graphHeight = 500;
    const videoWidth = 190;
    const videoHeight = 190;

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
            console.log("selected", curNode);
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

            if (figure === "distribution") {
                drawScatterplot(classificationData, chart.nodes, curNode);
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
                let classificationData = {
                    accuracy: resp.data.accuracy,
                    matrix: resp.data.confusionMatrix,
                    states: resp.data.states,
                    dataPoints: resp.data.dataPoints
                }
                
                let node = nodes.find((n) => n.id === selectedNode?.id);
                setSelectedNode(node);

                if (figure === "confusion matrix") {
                    drawConfusionMatrix(classificationData);
                }
                else if (figure === "distribution") {
                    drawScatterplot(classificationData, nodes, node);
                }
                setClassificationData(classificationData);
            })
    };

    const drawConfusionMatrix = (classificationData) => {
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

        document.getElementById("graph-panel").innerHTML = "";
        let svg = d3.select("#graph-panel").append("svg")
            .attr("width", graphWidth)
            .attr("height", graphHeight);

        svg.append("text")
            .attr("x", graphWidth / 2)
            .attr("y", accuracyOffset)
            .attr("font-size", 20)
            .style("text-anchor", "middle")
            .text("Average Accuracy: " + classificationData.accuracy);

        // svg.append("text")
        //     .attr("x", 15)
        //     .attr("y", offsetY + legendHeight / 2)
        //     .attr("font-size", 20)
        //     .style("text-anchor", "middle")
        //     .text("True Labels")
        //     .style("writing-mode", "tb")
        //     .style("glyph-orientation-vertical", 90)

        let cellSize = legendHeight / classificationData.matrix.length;
        let colorScale = d3.scaleLinear()
            .domain([0, 0.5, 1])
            .range(["#FFFFDD", "#3E9583", "#1F2D86"])

        svg.selectAll('rect')
            .data(classificationData.matrix)
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
            .data(classificationData.matrix)
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
            .data(classificationData.states)
            .enter()
            .append('text')
            .text((d) => d)
            .attr("x", offsetX - labelOffset)
            .attr("y", (d, i) => offsetY + i * cellSize + cellSize / 2)
            .attr("alignment-baseline", "middle")
            .attr("text-anchor", "end")
            .attr("font-size", 16)

        svg.selectAll('.predict-states')
            .data(classificationData.states)
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

    const drawScatterplot = (classificationData, nodes, selectedNode) => {
        let data = classificationData.dataPoints;
        let margin = 10;

        document.getElementById("graph-panel").innerHTML = "";
        let svg = d3.select("#graph-panel").append("svg")
            .attr("width", graphWidth)
            .attr("height", graphHeight);

        let xScaler = d3.scaleLinear()
            .domain([d3.min(data, d => d.x), d3.max(data, d => d.x)])
            .range([margin, graphWidth - margin]);

        let yScaler = d3.scaleLinear()
            .domain([d3.min(data, d => d.y), d3.max(data, d => d.y)])
            .range([margin, graphHeight - margin]);

        svg.append("g")
            .attr("transform", `translate(0,${graphHeight / 2 - margin})`)
            .call(d3.axisBottom(xScaler))

        svg.append("g")
            .attr("transform", `translate(${graphWidth / 2 - margin},0)`)
            .call(d3.axisLeft(yScaler))

        const customColors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
            '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8',
            '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000'];
        const color = d3.scaleOrdinal()
            .domain(data.map(d => d.label))
            .range(customColors);

        if (selectedNode) {
            svg.append("g")
                .attr("fill", "none")
                .selectAll("circle")
                .data(data)
                .join("circle")
                .attr("fill", d => {
                    if (d.label === selectedNode.id) {
                        return "blue";
                    }
                    else if (selectedNode.parentNode) {
                        let parent = nodes.find((n) => n.id === selectedNode.parentNode);
                        if (parent.data.children.includes(d.label)) {
                            return "skyblue";
                        }
                        else {
                            return "lightgrey";
                        }
                    }
                    else {
                        return "lightgrey";
                    }
                })
                .attr("transform", d => `translate(${xScaler(d.x)},${yScaler(d.y)})`)
                .attr("r", 3);
        }
        else {
            svg.append("g")
                .attr("fill", "none")
                .selectAll("circle")
                .data(data)
                .join("circle")
                .attr("fill", d => color(d.label))
                .attr("transform", d => `translate(${xScaler(d.x)},${yScaler(d.y)})`)
                .attr("r", 3);
        }
    };

    const handleClickSelect = (type) => {
        setFigure(type);
        if (type === "confusion matrix") {
            drawConfusionMatrix(classificationData);
        }
        else if (type === "distribution") {
            drawScatterplot(classificationData, chart.nodes, selectedNode);
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
            <div id="graph-panel">
                <Skeleton className="m-auto" variant="rectangular" animation="wave" width={graphWidth} height={graphHeight} />
            </div>

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