import { React, useState, useEffect } from "react";
import axios from "axios";
import { v4 as uuidv4} from "uuid";
import { Link } from 'react-router-dom';
import Typography from '@mui/material/Typography';
import { Button } from "@mui/material";
import "./home.css"

export default function Home(props) {
    let [boards, setBoards] = useState([]);

    useEffect(() => {
        // Side effect code goes here
        // It will run after the component renders
        getBoards();
        // Optional cleanup function
        return () => {
            // Cleanup code goes here
            // It will run before the component is removed from the DOM
        };
    }, []); // Dependency array

    function getBoards() {
        axios
            .get(window.BACKEND_ADDRESS + "/boards")
            .then((resp) => {
                console.log("Get boards from db", resp);
                setBoards(resp.data)
            })
    };

    function createBoard() {
        let newId = uuidv4();
        let curboard = {
            _id: newId
        };

        axios
            .post(window.BACKEND_ADDRESS + "/board", curboard)
            .then(response => {
                console.log("successed with response message:", response);
                window.location.href = 'board/' + newId;
            })
            .catch(error => {
                console.log("failed with error message:", error);
            });
    }

    return (
        <div className="home">
            <Typography variant="h1" gutterBottom>
                IoT Auditor
            </Typography>
            <Button variant="contained" onClick={createBoard}>Start Sensing</Button>
            <div className="storyexamplecontainer">
                <div className="leancontainer">
                    <p className="storyexamples">Boards</p>
                    <div className="row">
                        <div className="col-sm-12 text-start">
                            <div className='homeBoardList'>
                                {boards.map(board => (
                                    <div className='homeBoardStory' key={`row-${board._id}`}>
                                        <Link className='homeStoryLink'
                                            to={`board/${board._id}`}>
                                            {board.title}
                                        </Link>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}