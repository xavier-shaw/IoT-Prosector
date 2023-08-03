import React from "react";
import MenuBar from "../components/MenuBar";
import { Helmet } from 'react-helmet';
import "./board.css";
import { useState } from "react";
import { useEffect } from "react";
import NodeChart from "../components/NodeChart";
import TimelineChart from "../components/TimelineChart";

export default function Board(props) {
    let [board, setBoard] = useState({});

    useEffect(() => {
        setBoard({title: "untitled"});
    }, []);

    const handleTitleFocusOut = (titleText) => {
        console.log("Update board title:", titleText);
        setBoard(prevBoard => ({ ...prevBoard, title: titleText }));
    };

    const startSensing = () => {
        // TODO: continue fetching senser's data
    };

    return (
        <div className="board-div">
            <Helmet>
                <title>{board.title}</title>
            </Helmet>
            <MenuBar title={board.title} onTitleChange={handleTitleFocusOut} runProgram={startSensing}/>
            <div className="main-board-div">
                <div className="top-side-div">
                    <NodeChart/>
                </div>
                <div className="bottom-side-div">
                    <TimelineChart/>
                </div>
            </div>
        </div>
    )
}