var mongoose = require('mongoose');
var Schema = mongoose.Schema;

var SharedSchema = new Schema({
  _id: {
    type: String,
  },
  name: {
    type: String,
  },
  value: {
    type: String,
  }
});

module.exports = mongoose.model('sharedvariables', SharedSchema);