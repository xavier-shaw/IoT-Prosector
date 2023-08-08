import { Chip, TextField } from '@mui/material';
import React, { useEffect, useState } from 'react';
import { getBezierPath, EdgeLabelRenderer, BaseEdge } from 'reactflow';

export default function ExploreEdge(props) {
    let { sourceX,
        sourceY,
        sourcePosition,
        targetX,
        targetY,
        targetPosition, 
        markerEnd,
        id, data } = props;

    const [edgePath, labelX, labelY] = getBezierPath({
        sourceX,
        sourceY,
        sourcePosition,
        targetX,
        targetY,
        targetPosition
    });

    return (
        <>
            <BaseEdge id={id} path={edgePath} markerEnd={markerEnd}/>
            <EdgeLabelRenderer>
                <div
                    id={id + "_label"}
                    style={{
                        position: 'absolute',
                        transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
                        fontSize: 10,
                        pointerEvents: 'all'
                    }}
                    className="nodrag nopan"
                >
                    <Chip label={data.label}/>
                </div>
            </EdgeLabelRenderer>
        </>
    );
};