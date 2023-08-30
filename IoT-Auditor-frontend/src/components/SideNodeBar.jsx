import { Chip } from '@mui/material';
import React from 'react';
import "./SideNodeBar.css";

export default function SideNodeBar(props) {
    const onDragStart = (event, nodeType) => {
        event.dataTransfer.setData('application/reactflow', nodeType);
        event.dataTransfer.effectAllowed = 'move';
    };

    return (
        <div className='side-bar-div'>
            {/* <div className='system-node-div' onDragStart={(event) => onDragStart(event, 'systemNode')} draggable>
                System Node
            </div> */}
            <div className='mode-node-div' onDragStart={(event) => onDragStart(event, 'modeNode')} draggable>
                Mode Node
            </div>
            {/* <div className='group-node-div' onDragStart={(event) => onDragStart(event, 'groupNode')} draggable>
                Group Node
            </div> */}
            {/* <div className='operation-node-div' onDragStart={(event) => onDragStart(event, 'stateNode')} draggable>
                Operation Node
            </div> */}
        </div>
    );
};
