var State = require("../models/stateModel");

exports.get_all_states_by_device = function (req, res) {
    State.find({ device: req.params.device }).then((states) => {
        res.json(states);
    })
        .catch((error) => {
            console.error("Error occurs:", error);
        });
};
