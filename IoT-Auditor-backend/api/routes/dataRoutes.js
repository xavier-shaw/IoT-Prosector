module.exports = function (app) {
    var data = require('../controllers/dataController');

    app.route('/api/datas/:device')
        .get(data.get_all_datas_by_device);
};

