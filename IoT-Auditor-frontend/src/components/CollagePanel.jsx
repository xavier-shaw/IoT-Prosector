import React, { forwardRef, useEffect, useImperativeHandle, useState } from "react";
import * as d3 from "d3";
import "./CollagePanel.css";
import { Grid } from "@mui/material";
import axios from "axios";

const CollagePanel = forwardRef((props, ref) => {
    let { board, chart, chartSelection } = props;
    const [selectedNode, setSelectedNode] = useState(null);
    const [selectedNodeVideo, setSelectedNodeVideo] = useState(null);
    const videoWidth = 250;
    const videoHeight = 250;

    useImperativeHandle(ref, () => ({
        classifyStates
    }));

    useEffect(() => {
        classifyStates();
    }, [chart]);

    useEffect(() => {
        if (chartSelection.nodes[0]?.type === "stateNode") {
            let node_id = chartSelection.nodes[0].id;
            let stateInfo = chart.nodes.find((e) => e.id === node_id);
            setSelectedNode(stateInfo);
            axios
                .get(window.BACKEND_ADDRESS + "/video/get/" + node_id)
                .then((resp) => {
                    setSelectedNodeVideo(resp.data[0].video);
                })
        }
    }, [chartSelection]);

    const classifyStates = () => {
        axios
            .post(window.HARDWARE_ADDRESS + "/classification", {
                    device: board.title,
                    nodes: chart.nodes
            })
            .then((resp) => {
                let acc = resp.data.accuracy;
                let matrix = resp.data.confusionMatrix;
                let states = resp.data.states;
                drawConfusionMatrix(matrix, states, acc);
            })
    };

    const drawConfusionMatrix = (confusionMatrix, states, accuracy) => {
        let width = 650;
        let height = 550;
        let titleOffset = 30;
        let accuracyOffset = 30;
        let matrixOffset = 10;
        let labelOffset = 5;
        let labelRotate = -20;
        let legendOffsetX = 20;
        let legendOffsetY = 0;
        let legendWidth = 15;
        let margin = 80;

        let offsetX = 120;
        let offsetY = titleOffset + accuracyOffset + matrixOffset;
        let legendHeight = height - offsetY - margin;

        document.getElementById("confusion-matrix").innerHTML = "";
        let svg = d3.select("#confusion-matrix").append("svg")
            .attr("width", width)
            .attr("height", height);

        svg.append("text")
            .attr("x", width / 2)
            .attr("y", titleOffset)
            .attr("font-size", 30)
            .style("text-anchor", "middle")
            .text("Confusion Matrix");

        svg.append("text")
            .attr("x", width / 2)
            .attr("y", titleOffset + accuracyOffset)
            .attr("font-size", 20)
            .style("text-anchor", "middle")
            .text("Average Accuracy: " + accuracy);

        svg.append("text")
            .attr("x", 15)
            .attr("y", offsetY + legendHeight / 2)
            .attr("font-size", 20)
            .style("text-anchor", "middle")
            .text("True Labels")
            .style("writing-mode", "tb")
            .style("glyph-orientation-vertical", 90)

        let cellSize = legendHeight / confusionMatrix.length;
        // let colorScale = d3.scaleSequential(d3.interpolateRgbBasis(["#FFFFDD", "#3E9583", "#1F2D86"]))
        //     .domain([0, 1])
        let colorScale = d3.scaleLinear()
            .domain([0, 0.5, 1])
            .range(["#FFFFDD", "#3E9583", "#1F2D86"])

        svg.selectAll('rect')
            .data(confusionMatrix)
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
            .data(confusionMatrix)
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
            .attr('dy', '.35em')
            .text(d => d);

        svg.selectAll('.true-states')
            .data(states)
            .enter()
            .append('text')
            .text((d) => d)
            .attr("x", offsetX - labelOffset)
            .attr("y", (d, i) => offsetY + i * cellSize + cellSize / 2)
            .attr("alignment-baseline", "middle")
            .attr("text-anchor", "end")
            .attr("font-size", 12)

        svg.selectAll('.predict-states')
            .data(states)
            .enter()
            .append('text')
            .text((d) => d)
            .attr("x", (d, i) => offsetX + i * cellSize + cellSize / 2)
            .attr("y", offsetY + legendHeight + labelOffset + 10)
            .attr("text-anchor", "end")
            .attr("font-size", 12)
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


    return (
        <div className="collage-panel-div">
            <div id="confusion-matrix"></div>

            {/* TODO: 1. Stepwise Debugging => State Info (video + timewise-next-state + related action / labeled action) 
                      2. Group Node and Split Node => change should be demonstrated by confusion matrix  
            */}
            <h3>State Information</h3>
            {selectedNode !== null && selectedNodeVideo != null &&
                <Grid container columnSpacing={1}>
                    <Grid item xs={4}>
                        <Grid container columnSpacing={1}>
                            <Grid item>
                                <h6>Label:</h6>
                                <h6>Action:</h6>
                                <h6>Previous:</h6>
                            </Grid>
                            <Grid item>
                                <h6>{selectedNode.data.label}</h6>
                                <h6>{selectedNode.data.action}</h6>
                                <h6>{selectedNode.data.prev ? chart.nodes.find((e) => e.id === selectedNode.data.prev).data.label : "NaN"}</h6>
                            </Grid>
                        </Grid>
                    </Grid>
                    <Grid item xs={8}>
                        <video src={selectedNodeVideo} controls width={videoWidth} height={videoHeight} />
                    </Grid>
                </Grid>
            }
        </div>
    )
})

export default CollagePanel;