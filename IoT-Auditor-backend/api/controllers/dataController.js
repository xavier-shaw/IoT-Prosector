var Data = require("../models/dataModel");

exports.get_all_datas_by_device = function (req, res) {
    Data.find({ device: req.params.device }).then((datas) => {
        res.json(datas);
    })
        .catch((error) => {
            console.error("Error occurs:", error);
        });
};