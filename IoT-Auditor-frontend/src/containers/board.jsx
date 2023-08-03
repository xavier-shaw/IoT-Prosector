import React from "react";
import MenuBar from "../components/MenuBar";
import { Helmet } from 'react-helmet';
import "./board.css";
import axios from "axios";
import { useState } from "react";
import { useEffect } from "react";
import { cloneDeep } from 'lodash'
import { useParams } from "react-router-dom";
import NodeChart from "../components/NodeChart";
import TimelineChart from "../components/TimelineChart";

export default function Board(props) {
    let params = useParams();
    let [board, setBoard] = useState({});

    useEffect(() => {
        axios
            .get(window.BACKEND_ADDRESS + "/boards/" + params.boardId)
            .then((resp) => {
                let board_data = resp.data[0];
                board_data.data = JSON.parse(board_data.data);
                console.log("board data", board_data);
                setBoard(board_data);
            });
        return () => {
        }
    }, [params.boardId]);

    const handleTitleFocusOut = (titleText) => {
        console.log("Update board title:", titleText);
        setBoard(prevBoard => ({ ...prevBoard, title: titleText }));
    };

    const onSave = () => {
        console.log("update board data", board);
        let board_cpy = cloneDeep(board);
        board_cpy["data"] = JSON.stringify(board["data"]);
        axios
            .post(window.BACKEND_ADDRESS + "/boards/saveBoard", { boardId: board._id, updates: board_cpy })
            .then((resp) => {
                console.log("update board", resp.data);
            })
    };

    const startSensing = () => {
        // TODO: continue fetching senser's data
        axios.
            get(window.BACKEND_ADDRESS + "/states/" + board.title)
            .then((resp) => {
                console.log("iot states", resp.data);
                let iotStates = resp.data;
                let boardData = board.data;
                let states = board.data.hasOwnProperty("statesDict")? board.data.statesDict: {};
                let transitions = board.data.hasOwnProperty("transitionsDict")? board.data.transitionsDict: {};
            
                for (const iotState of iotStates) {
                    if (!states.hasOwnProperty(iotState.idx)) {
                        let state = {
                            id: iotState.idx,
                            type: "stateNode",
                            time: iotState.time,
                            // TODO: refine the position inital layout
                            position: { x: 50 + 200 * (parseInt(iotState.idx)), y: 100 },
                            data: { label: "State " + iotState.idx }
                        };
                        states[iotState.idx] = state;

                        if (iotState.idx !== iotState.prev_idx) {
                            let transition = {
                                id: "e" + iotState.prev_idx + "-" + iotState.idx,
                                source: iotState.idx,
                                target: iotState.prev_idx,
                                label: "action"
                            };
                            transitions[transition.id] = transition;
                        }
                    }
                };

                boardData.statesDict = states;
                boardData.transitionsDict = transitions;
                setBoard((prevBoard) => ({...prevBoard, data: boardData}))
            })
    };

    return (
        <div className="board-div">
            <Helmet>
                <title>{board.title}</title>
            </Helmet>
            <MenuBar title={board.title} onSave={onSave} onTitleChange={handleTitleFocusOut} runProgram={startSensing} />
            <div className="main-board-div">
                <div className="top-side-div">
                    <NodeChart board={board} setBoard={setBoard}/>
                </div>
                <div className="bottom-side-div">
                    <TimelineChart />
                </div>
            </div>
        </div>
    )
}