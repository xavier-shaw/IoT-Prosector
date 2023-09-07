import React, { forwardRef, useEffect, useImperativeHandle, useState } from "react";
import * as d3 from "d3";
import "./CollagePanel.css";
import { Button, Grid, Skeleton } from "@mui/material";
import axios from "axios";
import { customColors, noneColor, siblingColor, stateColor } from "../shared/chartStyle";

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
        if (chartSelection?.type === "stateNode") {
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
                let edge_id = chart.edges.find((e) => e.originalSource === prev_id && e.originalTarget === node_id).id;
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
        else if (chartSelection === null || chartSelection.type === "semanticNode") {
            if (selectedNode) {
                hideHints(selectedNode, "all");
                setHint("");
            }
            setSelectedNode(null);
            
            if (figure === "distribution") {
                drawScatterplot(classificationData, chart.nodes, chartSelection);
            }
        }
    }, [chartSelection]);

    const setVideoById = (id, setStateFunction) => {
        axios
            .get(window.BACKEND_ADDRESS + "/video/get/" + id)
            .then((resp) => {
                console.log("video", resp.data);
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

    // ============================= Confusion Matrix ================================
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

    // ============================= Scatterplot ================================
    const drawScatterplot = (classificationData, nodes, selectedNode) => {
        let data = classificationData.dataPoints;
        let margin = 5;
        let cubeSize = 15;
        let seqSize = 30;
        let graphOffsetX = 20;
        let graphOffsetY = 30;

        document.getElementById("graph-panel").innerHTML = "";
        let svg = d3.select("#graph-panel").append("svg")
            .attr("width", graphWidth)
            .attr("height", graphHeight);

        const color = d3.scaleOrdinal()
            .domain(nodes.filter((n) => n.type === "stateNode").map(d => d.id))
            .range(customColors);
        
        let stateNodes = nodes.filter((n) => n.type === "stateNode");
        let legend = svg.append("g")
            .attr("transform", `translate(${margin}, ${margin})`)
        legend
            .selectAll("rect")
            .data(stateNodes)
            .join("rect")
            .attr("fill", d => color(d.id))
            .attr("width", cubeSize)
            .attr("height", cubeSize)
            .attr("transform", (d, i) => `translate(${i * (cubeSize + seqSize)}, 0)`)
        legend
            .selectAll("text")
            .data(stateNodes)
            .join("text")
            .text(d => d.data.label.slice(0, 3))
            .attr("transform", (d, i) => `translate(${i * (cubeSize + seqSize) + cubeSize}, 13)`)

        let xScaler = d3.scaleLinear()
            .domain([d3.min(data, d => d.x), d3.max(data, d => d.x)])
            .range([graphOffsetX, graphWidth - graphOffsetX]);

        let yScaler = d3.scaleLinear()
            .domain([d3.min(data, d => d.y), d3.max(data, d => d.y)])
            .range([graphOffsetY, graphHeight - graphOffsetY]);

        svg.append("g")
            .attr("transform", `translate(0,${graphHeight / 2 - graphOffsetY})`)
            .call(d3.axisBottom(xScaler))

        svg.append("g")
            .attr("transform", `translate(${graphWidth / 2 - graphOffsetX},0)`)
            .call(d3.axisLeft(yScaler))

        if (selectedNode?.type === "stateNode") {
            svg.append("g")
                .attr("fill", "none")
                .selectAll("circle")
                .data(data)
                .join("circle")
                .attr("fill", d => {
                    if (d.label === selectedNode.id) {
                        return color(d.label);
                    }
                    else if (selectedNode.parentNode) {
                        let parent = nodes.find((n) => n.id === selectedNode.parentNode);
                        if (parent.data.children.includes(d.label)) {
                            return color(d.label);
                        }
                        else {
                            return noneColor;
                        }
                    }
                    else {
                        return noneColor;
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
                {prevNode && prevNodeVideo && selectedNode &&
                    <div>
                        <h5>Previous State</h5>
                        <h6>{prevNode.data.label}</h6>
                        <video src={prevNodeVideo} controls width={videoWidth} height={videoHeight} />
                    </div>
                }
                {prevNode && actionVideo && selectedNode &&
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