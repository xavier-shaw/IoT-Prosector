import React, { useEffect, useRef } from "react";
import * as d3 from 'd3';

export default function TimelineChart(props) {
    const { totalStates } = props;
    const width = 1000;
    const height = 200;
    const offsetX = 40;
    const offsetY = 20;

    useEffect(() => {
        document.getElementById("timeline-container").innerHTML = "";

        let svg = d3.select("#timeline-container")
            .append("svg")
            .attr('width', width)
            .attr('height', height)
            .style('border', '1px solid lightgray');

        const baseTime = totalStates[0].time;

        // Scales
        const xScale = d3.scaleLinear()
            .domain([0, d3.max(totalStates, s => (s.time - baseTime))])
            .range([offsetX, width - 100]);

        const yScale = d3.scalePoint()
            .domain(totalStates.map(s => s.data.label))
            .range([height - offsetY, 0])
            .padding(0.5);

        // Line generator
        const line = d3.line()
            .x(s => xScale((s.time - baseTime)))
            .y(s => yScale(s.data.label));

        // Append the line to the SVG
        svg.append("path")
            .datum(totalStates)
            .attr("fill", "none")
            .attr("stroke", "black")
            .attr("stroke-width", 1.5)
            .attr("d", line);

        // Create circles for the timeline points
        svg.selectAll('circle')
            .data(totalStates)
            .enter()
            .append('circle')
            .attr('cx', s => xScale((s.time - baseTime)))
            .attr('cy', s => yScale(s.data.label))
            .attr('r', 5)
            .attr('fill', 'blue');

        // Create x-axis
        svg.append('g')
            .attr('transform', `translate(0, ${height - offsetY})`)
            .call(d3.axisBottom(xScale).ticks(totalStates.length));

        // Create y-axis
        svg.append('g')
            .attr('transform', `translate(${offsetX}, 0)`)
            .call(d3.axisLeft(yScale).ticks(totalStates.length));

    }, [totalStates]);

    return (
        <div id="timeline-container">
        </div>
    )
}