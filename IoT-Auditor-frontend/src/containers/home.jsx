import { React, useState, useEffect } from "react";
import Typography from '@mui/material/Typography';
import { Button } from "@mui/material";
import "./home.css"

export default function Home(props) {
    let [boards, setBoards] = useState([]);

    useEffect(() => {
        // Side effect code goes here
        // It will run after the component renders
        // getBoards();
        // Optional cleanup function
        return () => {
            // Cleanup code goes here
            // It will run before the component is removed from the DOM
        };
    }, []); // Dependency array

    function getBoards() {
        axios
            .get(window.BACKEND_ADDRESS + "/boards/example")
            .then((resp) => {
                console.log("Get example boards from db", resp);
                setBoards(resp.data)
            })
    };

    function createBoard() {
        window.location.href = 'board/';
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