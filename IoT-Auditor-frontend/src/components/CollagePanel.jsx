import React, { useEffect, useState } from "react";
import * as d3 from "d3";
import "./CollagePanel.css";
import { Grid } from "@mui/material";
import axios from "axios";

export default function CollagePanel(props) {
    let { board, chart, chartSelection } = props;
    const [selectedNode, setSelectedNode] = useState(null);
    const [selectedNodeVideo, setSelectedNodeVideo] = useState(null);
    const [confusionMatrix, setConfusionMatrix] = useState([]);
    const [accuracy, setAccuracy] = useState(null);
    const videoWidth = 300;
    const videoHeight = 300;
    const matrixWidth = 450;
    const matrixHeight = 450;

    useEffect(() => {
        console.log("here", chart);
        classifyStates();
    }, []);

    useEffect(() => {
        console.log(chartSelection)
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
            .get(window.HARDWARE_ADDRESS + "/classification", {
                params: {
                    device: board.title
                }
            })
            .then((resp) => {
                console.log(resp);
                let acc = resp.data.accuracy;
                let matrix = resp.data.confusionMatrix;
                setAccuracy(acc);
                setConfusionMatrix(matrix);
                drawConfusionMatrix(matrix);
            })
    };

    const drawConfusionMatrix = (confusionMatrix) => {
        document.getElementById("confusion-matrix").innerHTML = "";
        let svg = d3.select("#confusion-matrix").append("svg")
            .attr("width", matrixWidth)
            .attr("height", matrixHeight);

        let cellSize = matrixWidth / confusionMatrix.length;
        let colorScale = d3.scaleSequential(d3.interpolateRgbBasis(["#c9ebf6", "#4dc1f6", "#2251c8"]))
            .domain([0, 1])

        svg.selectAll('rect')
            .data(confusionMatrix)
            .enter().append('g')
            .attr("transform", function (d, i) {
                return "translate(0, " + i * cellSize + ")"
            })
            .selectAll('rect')
            .data(d => d)
            .enter().append('rect')
            .attr('x', (d, i) => i * cellSize)
            .attr('y', (d, i) => 0)
            .attr('width', cellSize)
            .attr('height', cellSize)
            .attr('fill', (d) => colorScale(d))
            .attr('stroke', 'black');

        
        // svg.selectAll('text')
        //     .data(confusionMatrix)
        //     .enter().append('g')
        //     .attr("transform", function (d, i) {
        //         return "translate(0, " + i * cellSize + ")"
        //     })
        //     .selectAll('text')
        //     .data(d => d)
        //     .enter().append('text')
        //     .attr('x', (d, i) => i * cellSize + cellSize / 2)
        //     .attr('y', (d, i) => cellSize / 2)
        //     .attr('text-anchor', 'middle')
        //     .attr('dy', '.35em')
        //     .text(d => d);
    };


    return (
        <div className="collage-panel-div">
            <h3>Confusion Matrix of States</h3>
            {accuracy && <h5>Accuracy: {accuracy}</h5>}
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
}