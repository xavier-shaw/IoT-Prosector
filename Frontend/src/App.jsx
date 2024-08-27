import { Routes, Route, BrowserRouter as Router } from "react-router-dom";
import routes from "./shared/routes";
import './App.css'
import Home from "./containers/home";
import Board from "./containers/board";

function App() {

  return (
    <Router>
      <Routes>
        <Route exact path={routes.home} element={<Home />}></Route>
        <Route path={routes.board} element={<Board />}></Route>
      </Routes>
    </Router>
  )
}

export default App
