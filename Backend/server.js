require('dotenv').config();

var express = require('express'),
    path = require('path'),
    app = express(),
    cors = require('cors'),
    port = process.env.PORT || 9990,
    mongoose = require('mongoose'),
    bodyParser = require('body-parser');

// mongoose instance connection url connection
mongoose.Promise = global.Promise;
mongoose.connect(`mongodb+srv://${process.env.NAME}:${process.env.PASSWORD}@${process.env.CLUSTER}/${process.env.DB_NAME}`, {
    useNewUrlParser: true,
    useUnifiedTopology: true
})
.then(() => {
  console.log("connected to database - iotdb.")
})
.catch(err => console.error("connect to db failed.", err));

app.use(cors());

app.use(bodyParser.urlencoded({limit: '10mb', extended: true}));
app.use(bodyParser.json({limit: '10mb', extended: true}));

var boardRoutes = require('./api/routes/boardRoutes'); //importing route
boardRoutes(app); //register the route
var stateRoutes = require("./api/routes/stateRoutes");
stateRoutes(app);
var videoRoutes = require("./api/routes/videoRoutes");
videoRoutes(app);
var sharedRoutes = require("./api/routes/sharedRoutes");
sharedRoutes(app);
var predictRoutes = require("./api/routes/predictRoutes");
predictRoutes(app);

app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});

