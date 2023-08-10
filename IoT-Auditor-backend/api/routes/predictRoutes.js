module.exports = function (app) {
    var predict = require('../controllers/predictController');

    app.route('/api/predict/:device')
        .get(predict.get_predict_state);
};

