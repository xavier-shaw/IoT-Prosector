import React, { forwardRef, useEffect, useImperativeHandle, useState } from "react";
import * as d3 from "d3";
import "./CollagePanel.css";
import { Button, Divider, Grid, Skeleton } from "@mui/material";
import axios from "axios";
import { colorPalette, customColors, noneColor, siblingColor, stateColor } from "../shared/chartStyle";
import 'svg2pdf.js'
import { jsPDF } from "jspdf";

const CollagePanel = forwardRef((props, ref) => {
    let { board, chart, chartSelection, showHints, hideHints } = props;
    const [selectedNode, setSelectedNode] = useState(null);
    const [selectedNodeVideo, setSelectedNodeVideo] = useState(null);
    const [prevNode, setPrevNode] = useState(null);
    const [prevNodeVideo, setPrevNodeVideo] = useState(null);
    const [actionVideo, setActionVideo] = useState(null);
    const [figure, setFigure] = useState("correlation matrix");
    const [classificationData, setClassificationData] = useState({});
    const [hint, setHint] = useState("");
    const graphWidth = 700;
    const graphHeight = 540;
    const videoWidth = 180;
    const videoHeight = 180;

    useImperativeHandle(ref, () => ({
        classifyStates,
    }));

    useEffect(() => {
        if (chartSelection?.type === "stateNode") {
            if (selectedNode) {
                hideHints(selectedNode, "all");
                setHint("");
            }
            console.log("selected node", chartSelection)
            let node_id = chartSelection.id;
            let prev_id = chartSelection.data.prev;
            let curNode = chart.nodes.find((e) => e.id === node_id);
            setSelectedNode(curNode);
            setVideoById(node_id, setSelectedNodeVideo);
            if (prev_id && curNode.data.status !== "base state") {
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
                setStateFunction(resp.data[0].video);
            })
    }

    const classifyStates = (nodes) => {
        console.log("classify", nodes)
        axios
            .post(window.HARDWARE_ADDRESS + "/classification", {
                device: board.title,
                nodes: nodes
            })
            .then((resp) => {
                let classificationData = {
                    matrix: resp.data.matrix,
                    clusters: resp.data.clusters,
                    groups: resp.data.groups,
                    dataPoints: resp.data.data_points,
                    dataLabels: resp.data.data_labels
                }
                console.log("classification result", classificationData)
                let node = nodes.find((n) => n.id === selectedNode?.id);
                setSelectedNode(node);

                if (figure === "correlation matrix") {
                    drawCorrelationMatrix(classificationData);
                }
                else if (figure === "distribution") {
                    drawScatterplot(classificationData, nodes, node);
                }
                setClassificationData(classificationData);
            })
    };

    // ============================= correlation matrix ================================
    const drawCorrelationMatrix = (classificationData) => {
        let matrixOffset = 8;
        let labelOffset = 10;
        let labelRotate = -30;
        let legendOffsetX = 5;
        let legendOffsetY = 0;
        let legendWidth = 15;
        let legendTickWidth = 40;
        let margin = 150;
        let offsetX = 120;
        let offsetY = matrixOffset;
        let matrixWidth = graphWidth - offsetX - legendWidth - legendOffsetX - legendTickWidth;
        let matrixHeight = graphHeight - offsetY - margin;

        document.getElementById("graph-panel").innerHTML = "";
        let svg = d3.select("#graph-panel").append("svg")
            .attr("id", "svg")
            .attr("width", graphWidth)
            .attr("height", graphHeight);

        svg.append("text")
            .style("font-size", 22)
            .style("font-family", "Times New Roman")
            .style("text-anchor", "middle")
            .text("Sensing Model")
            .attr("transform", "translate(" + 15 + ", " + (offsetY + matrixHeight / 2) + ") rotate(-90)")

        svg.append("text")
            .attr("x", offsetX + matrixWidth / 2)
            .attr("y", graphHeight - 10)
            .style("font-size", 22)
            .style("font-family", "Times New Roman")
            .style("text-anchor", "middle")
            .text("Mental Model")
        // .style("writing-mode", "tb")
        // .style("glyph-orientation-vertical", 90)

        let cellWidth = matrixWidth / classificationData.matrix[0].length;
        let cellHeight = matrixHeight / classificationData.matrix.length;
        let colorScale = d3.scaleLinear()
            .domain([0, 0.5, 1])
            .range(["#FFFFDD", "#3E9583", "#1F2D86"])

        svg.selectAll('rect')
            .data(classificationData.matrix)
            .enter().append('g')
            .attr("transform", function (d, i) {
                return "translate(" + offsetX + ", " + (i * cellHeight + offsetY) + ")"
            })
            .selectAll('rect')
            .data(d => d)
            .enter().append('rect')
            .attr('x', (d, i) => i * cellWidth)
            .attr('y', (d, i) => 0)
            .attr('width', cellWidth)
            .attr('height', cellHeight)
            .attr('fill', (d) => colorScale(d))
            .attr('stroke', 'black')

        svg.selectAll('.states-text')
            .data(classificationData.matrix)
            .enter().append('g')
            .attr("transform", function (d, i) {
                return "translate(" + offsetX + ", " + (i * cellHeight + offsetY) + ")"
            })
            .selectAll('.state-text')
            .data(d => d)
            .enter().append('text')
            .attr('x', (d, i) => i * cellWidth + cellWidth / 2)
            .attr('y', (d, i) => cellHeight / 2)
            .attr('text-anchor', 'middle')
            .style("font-size", 20)
            .style("font-family", "Times New Roman")
            .attr('dy', '.35em')
            .text(d => d);

        svg.selectAll('.clusters')
            .data(classificationData.clusters)
            .enter()
            .append('text')
            .text((d) => d)
            .attr("x", offsetX - labelOffset)
            .attr("y", (d, i) => offsetY + i * cellHeight + cellHeight / 2)
            .attr("alignment-baseline", "middle")
            .attr("text-anchor", "end")
            .style("font-size", 20)
            .style("font-family", "Times New Roman")

        svg.selectAll('.groups')
            .data(classificationData.groups)
            .enter()
            .append('text')
            .text((d) => d)
            .attr("x", (d, i) => offsetX + i * cellWidth + cellWidth / 2 + 10)
            .attr("y", offsetY + matrixHeight + labelOffset + 5)
            .attr("text-anchor", "end")
            .style("font-size", 20)
            .style("font-family", "Times New Roman")
            .attr("transform", (d, i) => "rotate(" + labelRotate + " " + (offsetX + i * cellWidth + cellWidth / 2) + " " + (offsetY + matrixHeight + labelOffset + 10) + ")")

        //Calculate the variables for gradient
        var gradientScale = d3.scaleLinear()
            .domain([0, 1])
            .range([0, matrixHeight])
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
                return gradientScale(gradientPoints[i]) / matrixHeight;
            })
            .attr("stop-color", function (d, i) {
                return colorScale(gradientPoints[i]);
            });

        var legendsvg = svg.append("g")
            .attr("transform", "translate(" + (offsetX + matrixWidth + legendOffsetX) + "," + (offsetY + legendOffsetY) + ")");

        //Draw the Rectangle
        legendsvg.append("rect")
            .attr("x", 0)
            .attr("y", 0)
            .attr("width", legendWidth)
            .attr("height", matrixHeight)
            .style("fill", "url(#legend-traffic)");

        //Set scale for x-axis
        var xScale = d3.scaleLinear()
            .range([0, matrixHeight])
            .domain([1, 0]);

        //Define x-axis
        var xAxis = d3.axisRight(xScale)
            .ticks(5)

        legendsvg.append("g")
            .attr("transform", "translate(" + legendWidth + ",0)")
            .call(xAxis)
            .selectAll(".tick text")
            .style("font-size", 20)
            .style("font-family", "Times New Roman")
    };

    // ============================= Scatterplot ================================
    const drawScatterplot = (classificationData, nodes, selectedNode) => {
        let data = classificationData.dataPoints;
        let labels = classificationData.dataLabels;
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
        // .attr("transform", (d, i) => `translate(${i * (cubeSize + seqSize)}, 0)`);

        legendItems
            .append("text")
            .text(d => {
                if (d.type === "stateNode") {
                    return d.data.label.split(" ")[0];
                } else {
                    return d.data.label.split(" ")[1];
                }
            })
            .style("font-size", 20)
            .style("font-family", "Times New Roman")
            .attr("transform", `translate(${cubeSize}, 10)`);


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

        if (selectedNode?.type === "stateNode") {
            svg.append("g")
                .attr("fill", "none")
                .selectAll("circle")
                .data(data)
                .join("circle")
                .attr("fill", (d, i) => {
                    let label = labels[i]
                    if (label === selectedNode.id) {
                        return color(label);
                    }
                    else if (selectedNode.parentNode) {
                        let parent = nodes.find((n) => n.id === selectedNode.parentNode);
                        if (parent.data.children.includes(label)) {
                            return color(label);
                        }
                        else {
                            return noneColor;
                        }
                    }
                    else {
                        return noneColor;
                    }
                })
                .attr("transform", d => `translate(${xScaler(d[0])},${yScaler(d[1])})`)
                .attr("r", 7);
        }
        else {
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
        }
    };

    const handleClickSelect = (type) => {
        setFigure(type);
        if (type === "correlation matrix") {
            drawCorrelationMatrix(classificationData);
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

    const exportSVG = () => {
        const element = document.getElementById("svg");
        let svgString = new XMLSerializer().serializeToString(element);
        var blob = new Blob([svgString], { type: "image/svg+xml;chartset=utf-8" });
        var url = URL.createObjectURL(blob);
        var a = document.createElement("a");
        a.href = url;
        a.download = figure + ".svg";
        a.click()
    }

    return (
        <div className="collage-panel-div">
            <div className="select-panel">
                <Button size="small" variant={figure === "correlation matrix" ? "contained" : "outlined"} color="primary" onClick={() => handleClickSelect("correlation matrix")}>
                    Correlation Matrix
                </Button>
                <Button size="small" variant={figure === "distribution" ? "contained" : "outlined"} color="primary" onClick={() => handleClickSelect("distribution")}>
                    Distribution Scatterplot
                </Button>
                <Button size="small" variant="outlined" onClick={exportSVG}>Export SVG</Button>
            </div>
            <div id="graph-panel">
                <Skeleton className="m-auto" variant="rectangular" animation="wave" width={graphWidth} height={graphHeight} />
            </div>

            <Divider className="mt-2">
                <h5 style={{marginBottom: "0px"}}>State Information</h5>
            </Divider>

            <div className="state-info-div">
                {prevNode && prevNodeVideo && selectedNode &&
                    <div>
                        <h5 style={{ marginBottom: "5px" }}>Previous State</h5>
                        <h6 style={{ marginBottom: "5px" }}>{prevNode.data.label}</h6>
                        <video src={prevNodeVideo} controls width={videoWidth} height={videoHeight} />
                    </div>
                }
                {prevNode && actionVideo && selectedNode &&
                    <div>
                        <h5 style={{ marginBottom: "5px" }}>Action</h5>
                        <h6 style={{ marginBottom: "5px" }}>{selectedNode.data.action}</h6>
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
                        <h5 style={{ marginBottom: "5px" }}>Current State</h5>
                        <h6 style={{ marginBottom: "5px" }}>{selectedNode.data.label}</h6>
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