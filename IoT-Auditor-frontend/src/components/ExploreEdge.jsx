import { Chip, TextField } from '@mui/material';
import React, { useEffect, useState } from 'react';
import { getBezierPath, EdgeLabelRenderer, BaseEdge, getSmoothStepPath, useNodes } from 'reactflow';
import { getEdgeStyle } from '../shared/chartStyle';
import {
    getSmartEdge, svgDrawSmoothLinePath,
    svgDrawStraightLinePath
} from '@tisoap/react-flow-smart-edge'
import { MarkerType } from "reactflow";
const markerStyle = {
    type: MarkerType.ArrowClosed,
    width: 30,
    height: 30,
    color: '#FF0072',
}

export default function ExploreEdge(props) {
    let { sourceX,
        sourceY,
        sourcePosition,
        targetX,
        targetY,
        targetPosition,
        markerEnd,
        style,
        id, data } = props;

    const [edgePath, labelX, labelY] = getBezierPath({
        sourceX,
        sourceY,
        sourcePosition,
        targetX,
        targetY,
        targetPosition
    });

    const nodes = useNodes().filter((n) => !n.parentNode);
    const options = {
        gridRatio: 8,
        drawEdge: svgDrawSmoothLinePath
    }
    const getSmartEdgeResponse = getSmartEdge({
        sourcePosition,
        targetPosition,
        sourceX,
        sourceY,
        targetX,
        targetY,
        nodes: nodes,
        options: options
    })

    // If the value returned is null, it means "getSmartEdge" was unable to find
    // a valid path, and you should do something else instead
    const k = true;
    if (getSmartEdgeResponse === null || k) {
        return (
            <>
                <BaseEdge id={id} path={edgePath} markerEnd={markerStyle} style={style} />
                <EdgeLabelRenderer>
                    <div
                        id={id + "_label"}
                        style={getEdgeStyle(labelX, labelY)}
                        className="nodrag nopan"
                    >
                        <Chip color="secondary" label={data.label} />
                    </div>
                </EdgeLabelRenderer>
            </>
        );
    }
    else {
        const { edgeCenterX, edgeCenterY, svgPathString } = getSmartEdgeResponse;
        return (
            <>
                <BaseEdge id={id} path={svgPathString} markerEnd={markerStyle} style={style}/>
                <EdgeLabelRenderer>
                    <div
                        id={id + "_label"}
                        style={getEdgeStyle(edgeCenterX, edgeCenterY)}
                        className="nodrag nopan"
                    >
                        <Chip color='secondary' label={data.label} />
                    </div>
                </EdgeLabelRenderer>
            </>
        );
    }
};