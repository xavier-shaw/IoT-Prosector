var mongoose = require('mongoose');
var Schema = mongoose.Schema;

var BoardSchema = new Schema({
  _id: {
    type: String,
  },
  title: {
    type: String,
    default: 'untitled'
  },
  created_date: {
    type: Date,
    default: Date.now
  },
  data: {
    type: String,
    default: '{}',
  }
});

module.exports = mongoose.model('boards', BoardSchema);