require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');
const LinearRegression = require('./linear-regression');
const plot = require('node-remote-plot');

let { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower', 'weight', 'displacement'],
    labelColumns: ['mpg']
});

// console.log(features, labels, testFeatures, testLabels);

const regression = new LinearRegression(features, labels, {
    learningRate: 0.1,
    iterations: 100,
    batchSize: 1
});

regression.train();

console.log('m = ', regression.weights.get(1, 0), 'b = ', regression.weights.get(0, 0));
console.log(regression.test(testFeatures, testLabels));
console.log(regression.mseHistory);

console.log(regression.predict([
    [120, 2, 380],
    [135, 2.1, 420]
]).print());


plot({
    x: regression.bHistory,
    y: regression.mseHistory.reverse(),
    xLabel: 'Value of B',
    yLabel: 'MSE'
})