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
                        transform: `translate(-50%, -50%) translate(${labelX + (targetX - labelX) / 2}px,${labelY + (targetY - labelY) / 2}px)`,
                        fontSize: 10,
                        pointerEvents: 'all',
                        backgroundColor: "rgba(0, 0, 0, 0.08)",
                        borderRadius: 10,
                        zIndex: 4
                    }}
                    className="nodrag nopan"
                >
                    <Chip label={data.label}/>
                </div>
            </EdgeLabelRenderer>
        </>
    );
};