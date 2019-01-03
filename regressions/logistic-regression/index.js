require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');
const LogisticRegression = require('./logistic-regression');
const plot = require('node-remote-plot');


let { features, labels, testFeatures, testLabels } = loadCSV('../data/cars.csv', {
  shuffle: true,
  splitTest: 50,
  dataColumns: ['horsepower', 'weight', 'displacement'],
  labelColumns: ['passedemissions'],
  converters: {
    passedemissions: value => {
      return value === 'TRUE' ? 1 : 0;
    }
  }
});

const regression = new LogisticRegression(features, labels, {
  learningRate: 0.1,
  iterations: 3,
  batchSize: 10,
  decisionBoundary: 0.5
});

regression.train();
const r2 = regression.test(testFeatures, testLabels);
console.log('R2 is', r2);

console.log(regression.predict([
  [130, 307, 1.75]
]).print());

plot({
  x: regression.costHistory.reverse(),
  xLabel: 'Iteration #',
  yLabel: 'Cost'
});

/* const regression = new LinearRegression(features, labels, {
  learningRate: 0.1,
  iterations: 3,
  batchSize: 10
});

regression.train();
const r2 = regression.test(testFeatures, testLabels);

plot({
  x: regression.mseHistory.reverse(),
  xLabel: 'Iteration #',
  yLabel: 'Mean Squared Error'
});

console.log('R2 is', r2);

regression.predict([[120, 2, 380]]).print(); */
