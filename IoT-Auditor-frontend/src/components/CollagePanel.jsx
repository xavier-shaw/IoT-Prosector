import React, { useEffect } from "react";
import * as d3 from "d3";
import "./CollagePanel.css";

export default function CollagePanel(props) {

    const confusionMatrix = [
        [9, 2, 3, 4, 5, 6, 7],
        [1, 2, 3, 4, 5, 6, 7],
        [1, 2, 3, 4, 5, 6, 7],
        [1, 2, 3, 4, 5, 6, 7],
        [1, 2, 3, 4, 5, 6, 7],
        [1, 2, 3, 4, 5, 6, 7],
        [1, 2, 3, 4, 5, 6, 7],
    ];

    useEffect(() => {
        drawConfusionMatrix(confusionMatrix);
    }, [])

    const drawConfusionMatrix = (confusionMatrix) => {
        let width = 500, height = 500;
        document.getElementById("confusion-matrix").innerHTML = "";
        let svg = d3.select("#confusion-matrix").append("svg")
            .attr("width", width)
            .attr("height", height);

        let cellSize = width / confusionMatrix.length;
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
            .attr('fill', '#eee')
            .attr('stroke', 'black');

        svg.selectAll('text')
            .data(confusionMatrix)
            .enter().append('g')
            .attr("transform", function (d, i) {
                return "translate(0, " + i * cellSize + ")"
            })
            .selectAll('text')
            .data(d => d)
            .enter().append('text')
            .attr('x', (d, i) => i * cellSize + cellSize / 2)
            .attr('y', (d, i) => cellSize / 2)
            .attr('text-anchor', 'middle')
            .attr('dy', '.35em')
            .text(d => d);
    };


    return (
        <div className="collage-panel-div">
            <h3>Confusion Matrix of States</h3>
            <div id="confusion-matrix"></div>

            {/* TODO: 1. Stepwise Debugging => State Info (video + timewise-next-state + related action/ labeled action) 
                      2. Group Node and Split Node => change should be demonstrated by confusion matrix  
            */}

        </div>
    )
}