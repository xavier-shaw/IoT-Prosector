var SharedVariables = require("../models/sharedModel");

exports.start_sensing = function (req, res) {
    SharedVariables.findOneAndUpdate({ name: "sensing" }, { "value": "true" }).then((resp) => {
        if (resp) {
            res.json(resp)
        }
    })
        .catch((error) => {
            console.error("Error occurs:", error);
        });
};

exports.stop_sensing = function (req, res) {
    SharedVariables.findOneAndUpdate({ name: "sensing" }, { "value": "false" }).then((resp) => {
        if (resp) {
            res.json(resp)
        }
    })
        .catch((error) => {
            console.error("Error occurs:", error);
        });
};